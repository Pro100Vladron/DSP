
import numpy as np
import matplotlib.pyplot as plt
import skimage.io

# -----------------------------
# Общие параметры
# -----------------------------
nu1 = 20.0   # Гц
nu2 = 67.0   # Гц
duration = 1.0  # длительность сигнала (сек)

# Высокая частота дискретизации для построения "аналогового" сигнала
fs_high = 10000
t_cont = np.linspace(0, duration, int(fs_high * duration), endpoint=False)
x_cont = np.sin(2 * np.pi * nu1 * t_cont) + np.sin(2 * np.pi * nu2 * t_cont)

# ==============================================================
# ЗАДАНИЕ 1. Определить допустимые значения частоты дискретизации
# ==============================================================
f0 = 2 * max(nu1, nu2)
print("Задание 1:")
print(f"  ν1 = {nu1} Гц, ν2 = {nu2} Гц. Максимальная частота νmax = {max(nu1,nu2)} Гц")
print(f"  f0 = 2·νmax = {f0} Гц")
print("  Условие Найквиста: fs >= f0 = 134 Гц.")
print()

# ==============================================================
# ЗАДАНИЕ 2. Частоты дискретизации
# ==============================================================
f1 = f0 - 10   # 124 Гц
f2 = f0 + 10   # 144 Гц
f3 = f0 + 1    # 135 Гц
fs_list = [int(f1), int(f2), int(f3)]
print("Задание 2: частоты дискретизации =", fs_list)
print()


# ---------------------------
# Функция вычисления спектра (по формуле 2.9)
# ---------------------------
def compute_spectrum(x, fs, f_range=(-1.0, 2.0), num_points=2000):
    """
    Вычисляет спектр дискретного сигнала по определению (формула 2.9)
    с периодическим продолжением.
    """
    N = len(x)
    freqs = np.linspace(f_range[0]*fs, f_range[1]*fs, num_points)  # частотная ось
    X = []

    for f in freqs:
        expo = np.exp(-1j * 2 * np.pi * f * np.arange(N) / fs)
        X.append(np.sum(x * expo))

    return freqs, np.array(X)


# ==============================================================
# ЗАДАНИЕ 3. Построение дискретных сигналов и спектров
# ==============================================================
nu1, nu2 = 20.0, 67.0   # те же частоты, что в условии
duration = 1.0          # берём 1 секунду для дискретизации

for fs in fs_list:
    N = int(fs * duration)  # число отсчётов
    n = np.arange(N)
    t_samples = n / fs
    x_samples = np.sin(2*np.pi*nu1*t_samples) + np.sin(2*np.pi*nu2*t_samples)

    # -----------------------
    # Дискретный сигнал во времени
    # -----------------------
    plt.figure(figsize=(10, 4))
    plt.stem(t_samples, x_samples,
             linefmt='C1-', markerfmt='C1o', basefmt='C1-',
             label=f'Отсчёты (fs={fs} Гц)')
    plt.title(f'Дискретный сигнал x[n], fs={fs} Гц')
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда')
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------
    # Спектр по формуле (2.9)
    # -----------------------
    freqs, X = compute_spectrum(x_samples, fs, f_range=(-1, 2), num_points=2000)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, np.abs(X), label=f'fs={fs} Гц')
    plt.title(f'Амплитудный спектр |X(f)|, fs={fs} Гц')
    plt.xlabel('Частота, Гц')
    plt.ylabel('|X(f)|')
    plt.xlim(-fs, 2*fs)
    plt.grid(True)
    plt.legend()
    plt.show()


# ==============================================================
# ЗАДАНИЕ 4. Иллюстрация наложения частот
# ==============================================================
fs_alias = 100   # неверная частота (< f0)
N_alias = int(fs_alias * duration)
n_alias = np.arange(N_alias)
t_samples_alias = n_alias / fs_alias
x_samples_alias = np.sin(2 * np.pi * nu1 * t_samples_alias) + np.sin(2 * np.pi * nu2 * t_samples_alias)

# sinc-интерполяция (ряд Котельникова)
def kotelnikov_reconstruct(xn, fs, t_grid):
    n = np.arange(len(xn))
    S = np.sinc(fs * t_grid[:, None] - n[None, :])
    return S @ xn

# Восстановленный сигнал
x_rec_from_alias = kotelnikov_reconstruct(x_samples_alias, fs_alias, t_cont)

plt.figure(figsize=(10,5))
plt.plot(t_cont, x_cont, label='Оригинал x(t)')
plt.plot(t_cont, x_rec_from_alias, label='Восстановлен (fs = 100 Гц)')
plt.stem(t_samples_alias, x_samples_alias,
         linefmt='C3-', markerfmt='C3o', basefmt='C3-', label='Отсчёты')
plt.xlim(0, 0.2)
plt.title('Задание 4: Эффект aliasing')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)
plt.show()


# ==============================================================
# ЗАДАНИЕ 5. Прореживание и восстановление изображения
# ==============================================================
from skimage.transform import resize

fd = skimage.io.imread(r"C:\Users\vladr\OneDrive\Documents\Учеба МИЭТ\3 курс\ЦОС\Задание 1\var1.png", as_gray=True)
M, N = fd.shape

plt.figure()
plt.imshow(fd, cmap='gray')
plt.title("Исходное изображение")
plt.axis("off")
plt.show()

for k in [2, 3, 4]:
    # ---------------- Прореживание ----------------
    ff = fd[::k, ::k]   # прореженное изображение
    Mk, Nk = ff.shape

    # ---------------- Восстановление по Котельникову ----------------
    ColumnInd = np.arange(0, max(Mk, Nk))
    SincArray = np.zeros((max(M, N), max(Mk, Nk)))
    for j in range(max(M, N)):
        SincArray[j] = np.sinc(j / k - ColumnInd)

    F_sinc = SincArray[:M, :Mk] @ ff @ np.transpose(SincArray[:N, :Nk])

    # ---------------- Восстановление через resize (бикубическая интерполяция) ----------------
    F_resize = resize(ff, (M, N), order=2, mode='reflect', anti_aliasing=True)

    # ---------------- Графики ----------------
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(ff, cmap='gray')
    plt.title(f"Прореженное (k={k})")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(F_sinc, cmap='gray')
    plt.title(f"Восстановление sinc (k={k})")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(F_resize, cmap='gray')
    plt.title(f"Восстановление resize (k={k})")
    plt.axis("off")

    plt.show()

    
