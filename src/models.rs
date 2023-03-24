use tch::{IndexOp, nn, Tensor};
use tch::nn::{Init, Module, ModuleT};


#[derive(Debug)]
pub struct BigramLanguageModel {
    embedding: nn::Embedding,

}

impl LanguageModel for BigramLanguageModel {
    fn loss(&self, xs: &Tensor, ys: &Tensor) -> Tensor {
        let logits = self.embedding.forward(xs);
        let size = logits.size();

        let [b, t, c] = [size[0], size[1], size[2]];

        let logits_flat = logits.view([b*t, c]);
        let targets = ys.view([b*t]).totype(tch::Kind::Int64);

        logits_flat.cross_entropy_for_logits(&targets)
    }

    fn generate(&self, tokens: &Tensor, max_new_tokens: i32) -> Tensor {
        let mut new_tokens = tokens.copy();
        let n = tokens.size()[0];
        let mut token = tokens.i(n-1);
        for _ in 0..max_new_tokens {
            let logits = self.embedding.forward(&token);
            let probs = logits.softmax(-1, tch::Kind::Float);
            // println!("probs: {:?}", probs);
            let next_token = i64::from(probs.multinomial(1, true));
            // let next_token = probs.argmax(-1, true).item::<i64>();
            new_tokens = Tensor::cat(&[new_tokens, Tensor::of_slice(&[next_token])], 0);
            token = Tensor::of_slice(&[next_token]);
        }
        new_tokens
    }
}

impl ModuleT for BigramLanguageModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        // xs.shape() = [BATCH_SIZE, BLOCK_SIZE]
        self.embedding.forward(xs)
    }
}

impl BigramLanguageModel {
    pub fn new(vs: &nn::Path, vocab_size: i64, embedding_dim: i64) -> BigramLanguageModel {
        let embedding = nn::embedding(vs, vocab_size, embedding_dim, Default::default());

        BigramLanguageModel {
            embedding
        }
    }
}


pub trait LanguageModel {
    fn loss(&self, xs: &Tensor, ys: &Tensor) -> Tensor;
    fn generate(&self, tokens: &Tensor, max_new_tokens: i32) -> Tensor;
}

#[derive(Debug)]
pub struct LayerNorm1D {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f64,
}

impl ModuleT for LayerNorm1D {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xmean = xs.mean_dim(Some([-1].as_slice()), true, tch::Kind::Float);
        let xvar = xs.var_dim(Some([-1].as_slice()), true, true);
        let xhat = (xs - xmean) / (xvar + self.eps).sqrt();
        &self.gamma * xhat + &self.beta
    }
}

impl LayerNorm1D {
    fn new(vs: &nn::Path, size: i64) -> LayerNorm1D {
        // let gamma = vs.var("gamma", &[size], Init::Const(1.));
        let gamma = vs.ones("gamma", &[size]);
        let beta = vs.zeros("beta", &[size]);
        LayerNorm1D {
            gamma,
            beta,
            eps: 1e-5,
        }
    }
}

#[derive(Debug)]
struct Head {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    dropout: f64,
    tril: Tensor,
}

impl Head {
    fn new(vs: &nn::Path, n_embd: i64, head_size: i64, block_size: i64, dropout: f64) -> Head {
        let key = nn::linear(vs, n_embd, head_size, nn::LinearConfig { bias: false, ..Default::default() });
        let query = nn::linear(vs, n_embd, head_size, nn::LinearConfig { bias: false, ..Default::default() });
        let value = nn::linear(vs, n_embd, head_size, nn::LinearConfig { bias: false, ..Default::default() });
        let tril = Tensor::tril(&Tensor::ones(&[block_size, block_size], (tch::Kind::Float, tch::Device::Cpu)), 0);
        Head {
            key,
            query,
            value,
            dropout,
            tril,
        }
    }
}

impl ModuleT for Head {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let [b, t, c] = [size[0], size[1], size[2]];

        let k = self.key.forward(xs);
        let q = self.query.forward(xs);
        let v = self.value.forward(xs);

        let mut wei = q.matmul(&k.transpose(-2, -1));
        wei /= (k.size()[1] as f64).sqrt();
        wei = wei.masked_fill(&self.tril.eq(0).i((..t, ..t)), -1e9);

        let mut a = wei.softmax(-1, tch::Kind::Float);
        a = a.dropout(self.dropout, train);
        a.matmul(&v)
    }
}


#[derive(Debug)]
struct MuliHeadAttention {
    heads: Vec<Head>,
    out: nn::Linear,
    dropout: f64,
}

impl MuliHeadAttention {
    fn new(vs: &nn::Path, n_embd: i64, n_head: i64, head_size: i64, block_size: i64, dropout: f64) -> MuliHeadAttention {
        let mut heads = Vec::new();

        for i in 0..n_head {
            let head = Head::new(&(vs / format!("head{}", i)), n_embd, head_size, block_size, dropout);
            heads.push(head);
        }
        let out = nn::linear(vs, n_embd, n_embd, Default::default());
        MuliHeadAttention {
            heads,
            out,
            dropout,
        }
    }
}

impl ModuleT for MuliHeadAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut h = Vec::new();
        for head in &self.heads {
            h.push(head.forward_t(xs, train));
        }
        let h = Tensor::cat(&h, -1);
        let h = self.out.forward_t(&h, train);
        h.dropout(self.dropout, train)
    }
}


#[derive(Debug)]
struct FeedForward {
    net: nn::SequentialT
}

impl FeedForward {
    fn new(vs: &nn::Path, n_embd: i64, n_inner: i64, dropout: f64) -> FeedForward {
        let net = nn::seq_t()
            .add(nn::linear(vs, n_embd, n_inner, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(vs, n_inner, n_embd, Default::default()))
            .add_fn_t(move |xs, train| xs.dropout(dropout, train));

        FeedForward {
            net
        }
    }
}

impl ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        self.net.forward_t(xs, train)
    }
}


#[derive(Debug)]
struct Block {
    attn: MuliHeadAttention,
    ln1: LayerNorm1D,
    mlp: FeedForward,
    ln2: LayerNorm1D,
}

impl Block {
    fn new(vs: &nn::Path, n_embd: i64, n_head: i64, head_size: i64, block_size: i64, n_inner: i64, dropout: f64) -> Block {
        let attn = MuliHeadAttention::new(&(vs / "attn"), n_embd, n_head, head_size, block_size, dropout);
        let ln1 = LayerNorm1D::new(&(vs / "ln1"), n_embd);
        let mlp = FeedForward::new(&(vs / "mlp"), n_embd, n_inner, dropout);
        let ln2 = LayerNorm1D::new(&(vs / "ln2"), n_embd);
        Block {
            attn,
            ln1,
            mlp,
            ln2,
        }
    }
}

impl ModuleT for Block {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let h = xs + self.attn.forward_t(&self.ln1.forward_t(xs, train), train);
        let g = &h + self.mlp.forward_t(&self.ln2.forward_t(&h, train), train);
        g
    }
}


#[derive(Debug)]
pub struct AttnLanguageModel {
    token_emb: nn::Embedding,
    position_emb: nn::Embedding,
    blocks: nn::SequentialT,
    ln_f: LayerNorm1D,
    head: nn::Linear,
}

impl AttnLanguageModel {
    pub fn new(vs: &nn::Path, vocab_size: i64, n_embd: i64, n_head: i64, head_size: i64, block_size: i64, n_inner: i64, n_layer: i64, dropout: f64) -> AttnLanguageModel {
        let token_emb = nn::embedding(vs, vocab_size, n_embd, Default::default());
        let position_emb = nn::embedding(vs, block_size, n_embd, Default::default());
        let mut blocks = nn::seq_t();
        for i in 0..n_layer {
            let block = Block::new(&(vs / format!("block{}", i)), n_embd, n_head, head_size, block_size, n_inner, dropout);
            blocks = blocks.add(block);
        }
        let ln_f = LayerNorm1D::new(&(vs / "ln_f"), n_embd);
        let head = nn::linear(vs, n_embd, vocab_size, Default::default());
        AttnLanguageModel {
            token_emb,
            position_emb,
            blocks,
            ln_f,
            head,
        }
    }
}

impl ModuleT for AttnLanguageModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let size = xs.size();
        let [b, t] = [size[0], size[1]];

        let mut h = self.token_emb.forward_t(xs, train) + self.position_emb.forward_t(&Tensor::arange(t, (tch::Kind::Int64, tch::Device::Cpu)), train);
        h = h.dropout(0.1, train);
        h = self.blocks.forward_t(&h, train);
        h = self.ln_f.forward_t(&h, train);
        self.head.forward_t(&h, train)
    }
}

impl LanguageModel for AttnLanguageModel {
    fn loss(&self, xs: &Tensor, ys: &Tensor) -> Tensor {
        let logits = self.forward_t(xs, true);
        let size = logits.size();

        let [b, t, c] = [size[0], size[1], size[2]];

        let logits_flat = logits.view([b*t, c]);
        let targets = ys.view([b*t]).totype(tch::Kind::Int64);

        logits_flat.cross_entropy_for_logits(&targets)
    }

    fn generate(&self, tokens: &Tensor, max_new_tokens: i32) -> Tensor {
        // let mut new_tokens = tokens.copy();
        // let n = tokens.size()[0];
        // for _ in 0..max_new_tokens {
        //     let logits = self.embedding.forward(&token);
        //     let probs = logits.softmax(-1, tch::Kind::Float);
        //     // println!("probs: {:?}", probs);
        //     let next_token = i64::from(probs.multinomial(1, true));
        //     // let next_token = probs.argmax(-1, true).item::<i64>();
        //     new_tokens = Tensor::cat(&[new_tokens, Tensor::of_slice(&[next_token])], 0);
        //     token = Tensor::of_slice(&[next_token]);
        // }
        // new_tokens
        let n = tokens.size()[0];
        let mut tokens = tokens.copy();
        for _ in 0..max_new_tokens {
            let size = tokens.size();
            let [b, t] = [size[0], size[1]];

            let tokens_cond = if t > 32 {
                tokens.i((.., t - 32..))
            } else {
                tokens.copy()
            };

            // tokens = tokens.i((.., tokens.size()[1] - 32..));
            let logits = self.forward_t(&tokens_cond, false).i((.., -1));
            let probs = logits.softmax(-1, tch::Kind::Float).squeeze();
            // println!("probs: {:?}", probs);
            let next_token = probs.multinomial(1, false).unsqueeze(0);
            // println!("tokens: {:?}", tokens);
            // println!("next_token: {:?}", next_token);
            tokens = Tensor::cat(&[tokens, next_token], 1);
        }
        tokens
    }
}