use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

// model.layers.0.input_layernorm.weight
// model.layers.0.mlp.up_proj.weight
// model.layers.0.self_attn.v_proj.weight
// model.layers.1.mlp.down_proj.weight
// model.layers.0.self_attn.k_proj.weight
// model.layers.1.mlp.gate_proj.weight
// model.layers.0.self_attn.q_proj.weight
// model.layers.0.post_attention_layernorm.weight
// model.layers.1.post_attention_layernorm.weight
// model.layers.1.input_layernorm.weight
// model.layers.0.self_attn.o_proj.weight
// model.layers.1.self_attn.q_proj.weight
// model.layers.1.self_attn.o_proj.weight
// model.norm.weight
// model.layers.1.self_attn.k_proj.weight
// model.layers.1.self_attn.v_proj.weight
// lm_head.weight
// model.layers.1.mlp.up_proj.weight
// model.layers.0.mlp.gate_proj.weight
// model.layers.0.mlp.down_proj.weight
// embedding_table

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| {
            let namelist = safetensor.names();
            for n in namelist {
                println!("{}", n);
            }
            println!("{}", name);
            let tensorname = match name {
                "lm_head" => "lm_head.weight",
                "embedding_table" => "lm_head.weight",
                "rms_out_w" => "model.norm.weight",
                _ => "",
            };

            let tensorview = safetensor.tensor(tensorname).unwrap();
            let vec_u8 = tensorview.data().to_vec();
            let mut vec_f32 = Vec::<f32>::new();
            for i in 0..(vec_u8.len() / 4) {
                let mut v: [u8; 4] = [0, 0, 0, 0];
                for j in 0..4 {
                    v[j] = vec_u8[i*4 + j];
                }
                vec_f32.push(f32::from_le_bytes(v));
            }
            Tensor::new(vec_f32, &tensorview.shape().to_vec())
        };
        
        let get_tensors = |name: &str| {
            let mut ret = Vec::<Tensor<f32>>::new();
            let namelist = safetensor.names();

            let tensorname = match name {
                "rms_att_w" => "lm_head.weight",
                "embedding_table" => "lm_head.weight",
                "rms_out_w" => "model.norm.weight",
                _ => "",
            };

            for n in namelist {
                if n.ends_with(tensorname) {
                    ret.push(get_tensor(n));
                }
            }
            return ret;
        };

        LLamaParams {
            embedding_table: get_tensor("embedding_table"),
            rms_att_w: get_tensors("rms_att_w"),
            wq: get_tensors("wq"),
            wk: get_tensors("wk"),
            wv: get_tensors("wv"),
            wo: get_tensors("wo"),
            rms_ffn_w: get_tensors("rms_ffn_w"),
            w_up: get_tensors("w_up"),
            w_gate: get_tensors("w_gate"),
            w_down: get_tensors("w_down"),
            rms_out_w: get_tensor("rms_out_w"),
            lm_head: get_tensor("lm_head"),
        }
    }
}
