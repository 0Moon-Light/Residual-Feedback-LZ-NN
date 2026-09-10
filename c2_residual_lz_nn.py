import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

# Ultra-clean GitHub Dark Theme
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.edgecolor'] = '#30363d'
plt.rcParams['axes.linewidth'] = 1.5

class LZ77Engine:
    def __init__(self, window_size=256, max_match_len=16):
        self.window_size = window_size
        self.max_match_len = max_match_len

    def find_match(self, data, pos):
        start_win = max(0, pos - self.window_size)
        window = data[start_win:pos]
        best_len = 0
        best_dist = 0
        for l in range(1, min(self.max_match_len, len(data) - pos) + 1):
            substring = data[pos:pos+l]
            found_idx = -1
            for i in range(len(window) - l + 1):
                if window[i:i+l] == substring:
                    found_idx = i
            if found_idx != -1:
                best_len = l
                best_dist = len(window) - found_idx
            else:
                break
        return best_len, best_dist

class ResidualFeedbackLZNN:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size

    def predict_heads(self, S_t, context_bytes):
        is_match, match_len, match_dist, literal_run = S_t
        p_match = np.ones(self.vocab_size) * (0.05 / self.vocab_size)
        if is_match and len(context_bytes) >= match_dist:
            copied_byte = context_bytes[-match_dist]
            p_match[copied_byte] += 0.95
        p_match /= np.sum(p_match)

        p_literal = np.ones(self.vocab_size) / self.vocab_size
        if len(context_bytes) > 0:
            last_byte = context_bytes[-1]
            p_literal[(last_byte + 1) % self.vocab_size] += 0.6
            p_literal[last_byte] += 0.2
        p_literal /= np.sum(p_literal)

        p_gated = 0.4 * p_match + 0.6 * p_literal
        return np.vstack([p_match, p_literal, p_gated])

def run_simulation():
    np.random.seed(42)
    sample_text = b"THE_QUICK_BROWN_FOX_JUMPS_OVER_THE_LAZY_DOG_THE_QUICK_BROWN_FOX_HAS_REPETITIONS_THE_QUICK_BROWN_FOX_SAMPLE_DATA_TEXT_STREAM"
    data = list(sample_text)
    
    lz = LZ77Engine(window_size=64, max_match_len=8)
    model = ResidualFeedbackLZNN(vocab_size=256)
    
    literal_run = 0
    standard_nn_losses = []
    residual_lz_nn_losses = []
    kl_arbitrator_losses = []
    
    alpha, beta, gamma = 1.2, 0.8, 1.5
    
    for t in range(1, len(data)):
        target_byte = data[t]
        context = data[:t]
        match_len, match_dist = lz.find_match(data, t)
        is_match = 1 if match_len > 0 else 0
        literal_run = 0 if is_match else literal_run + 1
            
        S_t = (is_match, match_len, match_dist, literal_run)
        p_experts = model.predict_heads(S_t, context)
        p_consensus = np.mean(p_experts, axis=0)
        
        eps = 1e-12
        L_i = -np.log(p_experts[:, target_byte] + eps)
        V_i = (L_i - np.mean(L_i))**2
        A_i = np.zeros(3)
        for i in range(3):
            A_i[i] = -np.sum(p_experts[i] * np.log((p_experts[i] + eps) / (p_consensus + eps)))
            
        logits = -alpha * L_i - beta * V_i + gamma * A_i
        w_kl = softmax(logits)
        p_final_kl = np.dot(w_kl, p_experts)
        
        standard_nn_losses.append(-np.log(p_experts[1, target_byte] + eps))
        residual_lz_nn_losses.append(-np.log(p_experts[0 if is_match else 1, target_byte] + eps))
        kl_arbitrator_losses.append(-np.log(p_final_kl[target_byte] + eps))

    # ==========================================
    # CZYSTY, MODERN WYKRES (BEZ NORMALA TYTUŁÓW I NAZW PRODUKTU)
    # ==========================================
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), facecolor='#0d1117')

    for ax in axs:
        ax.set_facecolor('#161b22')
        ax.grid(True, linestyle='--', alpha=0.15, color='#8b949e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    cum_std = np.cumsum(standard_nn_losses)
    cum_res = np.cumsum(residual_lz_nn_losses)
    cum_kl = np.cumsum(kl_arbitrator_losses)
    
    # Subplot 1: Cumulative Loss
    axs[0].plot(cum_std, label='Standard NN (No LZ State)', color='#f85149', linewidth=2.5)
    axs[0].plot(cum_res, label='Hard-Switch LZ+NN', color='#d29922', linewidth=2.5)
    axs[0].plot(cum_kl, label='Residual-Feedback LZ+NN (KL-Arbitrator)', color='#2ea043', linewidth=3)
    axs[0].set_ylabel("Cumulative Loss (Nats)", color='#c9d1d9', fontweight='bold')
    axs[0].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', framealpha=0.9, loc='upper left')

    # Subplot 2: Instantaneous Loss
    axs[1].plot(standard_nn_losses, alpha=0.35, color='#f85149', label='Standard NN Loss')
    axs[1].plot(kl_arbitrator_losses, color='#2ea043', alpha=0.95, linewidth=1.8, label='Residual-Feedback Loss')
    axs[1].set_xlabel("Byte Position (t)", color='#c9d1d9', fontweight='bold')
    axs[1].set_ylabel("Per-Byte Cross-Entropy", color='#c9d1d9', fontweight='bold')
    axs[1].legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9', framealpha=0.9, loc='upper right')

    plt.tight_layout()
    out_img = "/home/moonlight/Pulpit/PHANTOMAI/0moon_light_c2_benchmark.png"
    plt.savefig(out_img, dpi=300, facecolor=fig.get_facecolor())
    print(f"Wygenerowano czysty wykres: {out_img}")

if __name__ == '__main__':
    run_simulation()
 
