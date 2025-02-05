r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 256
    hypers["seq_len"] = 128
    hypers["h_dim"] = 512
    hypers["n_layers"] = 2
    hypers["dropout"] = 0.125
    hypers["learn_rate"] = 0.001
    hypers["lr_sched_factor"] = 0.075
    hypers["lr_sched_patience"] = 0.8
    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    temperature = 0.7
    start_seq = "ACT I. SCENE 1"
    # ========================
    return start_seq, temperature


part1_q1 = """
We split the corpus into sequences instead of training on the whole text to improve memory efficiency, enable batch processing, and stabilize gradients.
Shorter sequences help the model learn local patterns effectively while avoiding issues like vanishing gradients.
This also speeds up training and better reflects real-world text processing scenarios.
"""

part1_q2 = r"""
**Your answer:**
RNN models retain memory through their hidden state, which gets updated at each training step.
The model predicts the next character using both the hidden state and the last character as input, allowing it to carry information across steps.
Due to its recurrent nature, it can generate sequences of any length.
However, since it is trained on fixed-length sequences, coherence may diminish over time.
"""

part1_q3 = r"""
**Your answer:**
We do not shuffle batches during training to maintain the sequential structure of the text.
This ensures that the model preserves context and learns long-term dependencies effectively.
Shuffling would break continuity, disrupting hidden states and making it harder for the model to generate coherent text.

"""

part1_q4 = r"""
**Your answer:**
Why do we lower the temperature for sampling (compared to the default of 1.0)?

Lowering the temperature reduces randomness in the model's output, making it more deterministic.
This is useful when we desire more predictable and coherent text generation. 

What happens when the temperature is very high and why?

At very high temperatures, the model's output becomes more random because the probability distribution over possible next tokens flattens.
This leads to more diverse but potentially less coherent text. 

What happens when the temperature is very low and why?

With very low temperatures, the model becomes highly deterministic, often choosing the most probable next token.
This can result in repetitive and less creative text, as the model is less likely to explore alternative word choices. 

"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers["batch_size"] = 96
    hypers["h_dim"] = 512
    hypers["z_dim"] = 64
    hypers["x_sigma2"] = 0.00025
    hypers["learn_rate"] = 0.00018
    hypers["betas"] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

The hyperparameter \( \sigma^2 \) (or `x_sigma2` in the code) in a Variational Autoencoder (VAE) sets the assumed variance of the data in the reconstruction term.
It effectively controls how "strictly" the model tries to match each input:

- **Lower \( \sigma^2 \)**: The model penalizes reconstruction errors more heavily, pushing it to generate outputs that closely match the training data.
This can lead to better fidelity in reconstructions but often a more constrained (less diverse) latent space.

- **Higher \( \sigma^2 \)**: The model becomes more tolerant of reconstruction errors, placing relatively greater emphasis on latent space regularization (via the KL divergence term).
This usually increases the diversity of generated samples at the cost of less precise reconstructions.
"""

part2_q2 = r"""
**Your answer:**

**VAE Loss Components**

- **Reconstruction Term:** Measures how accurately the decoder reproduces the original input. Lower errors here mean the model is effectively capturing core features of the data.
- **KL Divergence Term:** Regularizes the encoder's distribution \(q(z|x)\) to stay close to a chosen prior (usually a standard normal). Without this term, the encoder might place latent codes arbitrarily, which can hamper meaningful sampling and interpolation.

**Effect on Latent-Space Distribution**

Incorporating the KL term guides the learned latent distributions toward the prior, typically centering them near zero with unit variance. This regularization ensures that different data points share a common structure in the latent space, rather than lying in disjoint or degenerate regions.

**Benefit of the KL Term**

- **Better Sampling:** Because the latent distribution aligns with the prior, random draws in latent space decode into coherent outputs.
- **Smooth Interpolation:** Small steps in the latent space lead to gradual, understandable variations in the generated data, facilitating intuitive exploration of learned representations.
"""

part2_q3 = r"""
**Your answer:**

We want our generative model to assign high probability to the actual data, which is mathematically expressed as maximizing \(p(X)\).
This "evidence" represents the likelihood of observing the training data under the model's parameters.
Although directly maximizing \(p(X)\) is often intractable due to the necessary integration over latent variables, the VAE framework overcomes this by maximizing a lower bound on \(p(X)\), known as the ELBO.
This lower bound formulation ensures that the model learns to both reconstruct the data accurately and maintain a well-structured latent space.
"""

part2_q4 = r"""
**Your answer:**

We model the logarithm of the latent variance for practical and numerical reasons.
Exponentiating the output of the encoder ensures the resulting variance is always positive, which avoids dealing with strict non-negativity constraints.
Moreover, working in log space helps stabilize training, especially when variance values range widely, by keeping parameter updates on a manageable scale and preventing extremely large or small gradients.
"""

# Part 3 answers
def part3_gan_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======

    # ========================
    return hypers

part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**


"""

part3_q3 = r"""
**Your answer:**



"""



PART3_CUSTOM_DATA_URL = "https://github.com/AviaAvraham1/TempDatasets/raw/refs/heads/main/George_W_Bush2.zip"


def part4_transformer_encoder_hyperparams():
    hypers = dict(
        embed_dim = 0, 
        num_heads = 0,
        num_layers = 0,
        hidden_dim = 0,
        window_size = 0,
        droupout = 0.0,
        lr=0.0,
    )

    # TODO: Tweak the hyperparameters to train the transformer encoder.
    # ====== YOUR CODE: ======
    
    # ========================
    return hypers




part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**


"""


part4_q1 = r"""
**Your answer:**


"""

part4_q2 = r"""
**Your answer:**


"""


part4_q3= r"""
**Your answer:**


"""

part4_q4 = r"""
**Your answer:**


"""

part4_q5 = r"""
**Your answer:**


"""


# ==============
