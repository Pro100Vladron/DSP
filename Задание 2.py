import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Задание 1. Синтез случайного дискретного сигнала x(k) с равномерным распределением
# ============================================================
N = 512
low, high = -1, 1
x = np.random.uniform(low, high, size=N)
k = np.arange(N)

plt.figure(figsize=(10, 4))
plt.plot(k[:100], x[:100], linestyle='-', label='Сигнал')
plt.plot(k[:100], x[:100], 'ro', markersize=4, label='Отсчёты')  # красные точки
plt.title('Оригинальный сигнал x(k) (первые 100 отсчётов)')
plt.xlabel('k (отсчёт)')
plt.ylabel('x(k)')
plt.grid(True)
plt.legend()
plt.show()

# ============================================================
# Вспомогательная функция: равномерное квантование
# ============================================================
def uniform_quantize(x, bits, x_min=None, x_max=None):
    L = 2 ** bits
    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)
    step = (x_max - x_min) / (L - 1)
    indices = np.round((x - x_min) / (x_max - x_min) * (L - 1)).astype(int)
    indices = np.clip(indices, 0, L - 1)
    return x_min + indices * step

# ============================================================
# Задание 2. Равномерное квантование (1–8 бит)
# ============================================================
for bits in range(1, 9):
    xq = uniform_quantize(x, bits)

    plt.figure(figsize=(10, 4))
    edges = np.arange(N+1)
    plt.stairs(xq[:50], edges[:51], linewidth=2.0, label=f'Квантованный ({bits} бит)')
    plt.plot(k[:50], x[:50], 'ro', markersize=4, label='Отсчёты (оригинал)')
    plt.title(f'Квантованный сигнал ({bits} бит)')
    plt.xlabel('k (отсчёт)')
    plt.ylabel('x_q(k)')
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================================
# Задание 3. Ошибка квантования: выборочная и теоретическая
# ============================================================
bits_list = list(range(1, 9))
mse_sample = []
mse_theoretical = []

R = high - low
signal_power = np.mean(x**2)

for bits in bits_list:
    xq = uniform_quantize(x, bits, x_min=np.min(x), x_max=np.max(x))
    step = (np.max(x) - np.min(x)) / (2**bits - 1)

    e = x - xq
    mse_s = np.mean(e**2)
    mse_sample.append(mse_s)

    mse_th = (step**2) / 12.0
    mse_theoretical.append(mse_th)

plt.figure(figsize=(8,4))
plt.plot(bits_list, mse_sample, marker='o', label='Выборочное MSE')
plt.plot(bits_list, mse_theoretical, marker='s', linestyle='--', label='Теоретическая Δ²/12')
plt.yscale('log')
plt.grid(True, which='both', ls=':')
plt.xlabel('Число бит на отсчёт')
plt.ylabel('MSE')
plt.title('MSE квантования: выборочные и теоретические значения')
plt.xticks(bits_list)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Задание 4. SNR
# ============================================================
snr_sample_db = []
snr_theoretical_db = []

for i, bits in enumerate(bits_list):
    snr_sample_db.append(10 * np.log10(signal_power / mse_sample[i]))
    snr_theoretical_db.append(6.02 * bits + 1.76)  # классическая формула

plt.figure(figsize=(8,4))
plt.plot(bits_list, snr_sample_db, marker='o', label='SNR (выборочный)')
plt.plot(bits_list, snr_theoretical_db, marker='s', linestyle='--', label='SNR (теоретический)')
plt.grid(True)
plt.xlabel('Число бит на отсчёт')
plt.ylabel('SNR, dB')
plt.title('Зависимость SNR от числа бит')
plt.xticks(bits_list)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Задание 5. Сигнал y(k) ~ N(m, σ²)
# ============================================================
v = 1   # номер варианта
m = v
sigma = v + 1
N = 500

y = np.random.normal(loc=m, scale=sigma, size=N)
k = np.arange(N)

plt.figure(figsize=(10, 4))
plt.plot(k[:100], y[:100], linestyle='-', label='Сигнал')
plt.plot(k[:100], y[:100], 'ro', markersize=4, label='Отсчёты')
plt.title(f'Случайный сигнал y(k) ~ N({m}, {sigma}²) (первые 100 отсчётов)')
plt.xlabel('k (отсчёт)')
plt.ylabel('y(k)')
plt.grid(True)
plt.legend()
plt.show()

# ============================================================
# Задание 6. Параметры Ллойда-Макса
# ============================================================
lm_tables = {
    1: {"t": np.array([-np.inf, 0, np.inf]),
        "d": np.array([-0.7979, 0.7979])},
    2: {"t": np.array([-np.inf, -0.9816, 0, 0.9816, np.inf]),
        "d": np.array([-1.5104, -0.4528, 0.4528, 1.5104])},
    3: {"t": np.array([-np.inf, -1.7479, -1.05, -0.5005, 0, 0.5005, 1.05, 1.7479, np.inf]),
        "d": np.array([-2.1519, -1.3439, -0.756, -0.2451, 0.2451, 0.756, 1.3439, 2.1519])},
    4: {"t": np.array([-np.inf, -2.4008, -1.8435, -1.4371, -1.0993, -0.7995,
                       -0.5224, -0.2582, 0, 0.2582, 0.5224, 0.7995, 1.0993,
                       1.4371, 1.8435, 2.4008, np.inf]),
        "d": np.array([-2.7326, -2.069, -1.618, -1.2562, -0.9423, -0.6568,
                       -0.388, -0.1284, 0.1284, 0.388, 0.6568, 0.9423,
                       1.2562, 1.618, 2.069, 2.7326])}
}

# ============================================================
# Задания 7–8. Оптимальное квантование (Ллойд–Макс)
# ============================================================
mse_lm = []
snr_lm = []

for bits in range(1, 5):
    t_scaled = lm_tables[bits]["t"] * sigma + m
    d_scaled = lm_tables[bits]["d"] * sigma + m

    yq = np.empty_like(y)
    for i in range(len(d_scaled)):
        left = t_scaled[i]
        right = t_scaled[i+1]
        mask = (y > left) & (y <= right)
        yq[mask] = d_scaled[i]

    e = y - yq
    mse = np.mean(e**2)
    signal_power = np.mean(y**2)
    snr_db = 10 * np.log10(signal_power / mse)

    mse_lm.append(mse)
    snr_lm.append(snr_db)

    plt.figure(figsize=(10, 4))
    plt.step(k[:50], yq[:50], where='mid', label=f'Ллойд–Макс ({bits} бит)')
    plt.plot(k[:50], y[:50], 'ro', markersize=4, label='Отсчёты (оригинал)')
    plt.title(f'Оптимальное квантование (Ллойд–Макс), {bits} бит')
    plt.xlabel('k (отсчёт)')
    plt.ylabel('y(k), yq(k)')
    plt.grid(True)
    plt.legend()
    plt.show()

# ============================================================
# Задание 9. Равномерное квантование сигнала y(k)
# ============================================================
mse_uni = []
snr_uni = []

for bits in range(1, 5):
    yq = uniform_quantize(y, bits)
    e = y - yq
    mse = np.mean(e**2)
    signal_power = np.mean(y**2)
    snr_db = 10 * np.log10(signal_power / mse)

    mse_uni.append(mse)
    snr_uni.append(snr_db)

# Сравнение ошибок
bits_range = np.arange(1, 5)

plt.figure(figsize=(10, 5))
plt.semilogy(bits_range, mse_lm, 'ro-', label='Lloyd–Max')
plt.semilogy(bits_range, mse_uni, 'bs--', label='Равномерное')
plt.title('Ошибка квантования (MSE)')
plt.xlabel('Число бит на отсчёт')
plt.ylabel('MSE (лог. шкала)')
plt.grid(True, which="both")
plt.legend()
plt.show()

# Сравнение SNR
plt.figure(figsize=(10, 5))
plt.plot(bits_range, snr_lm, 'ro-', label='Lloyd–Max')
plt.plot(bits_range, snr_uni, 'bs--', label='Равномерное')
plt.title('SNR (дБ)')
plt.xlabel('Число бит на отсчёт')
plt.ylabel('SNR (дБ)')
plt.grid(True)
plt.legend()
plt.show()
