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
# ЗАДАНИЕ 2. Выполнить дискретизацию с заданными fs
# ==============================================================
f1 = f0 - 10   # 124 Гц
f2 = f0 + 10   # 144 Гц
f3 = f0 + 1    # 135 Гц
fs_list = [int(f1), int(f2), int(f3)]
print("Задание 2: частоты дискретизации =", fs_list)
print()

# ==============================================================
# ЗАДАНИЕ 3. Построить дискретные сигналы и их спектры (через FFT)
# ==============================================================

for fs in fs_list:
    N = int(fs * duration)  # число отсчётов
    n = np.arange(N)
    t_samples = n / fs
    x_samples = np.sin(2 * np.pi * nu1 * t_samples) + np.sin(2 * np.pi * nu2 * t_samples)

    # FFT с нулями (для повышения разрешения)
    Xf = np.fft.fft(x_samples, n=4096)
    freqs = np.fft.fftfreq(len(Xf), d=1/fs)

    # смещаем спектр так, чтобы он шёл от отрицательных к положительным частотам
    freqs_shifted = np.fft.fftshift(freqs)
    Xf_shifted = np.fft.fftshift(Xf)

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
    # Амплитудный спектр
    # -----------------------
    plt.figure(figsize=(10, 4))
    plt.plot(freqs_shifted, np.abs(Xf_shifted), label=f'fs={fs} Гц')
    plt.title(f'Амплитудный спектр |X(f)|, fs={fs} Гц')
    plt.xlabel('Частота, Гц')
    plt.ylabel('|X(f)|')
    plt.xlim(-fs, 2*fs)   # полоса по условию: [-fs, 2fs]
    plt.grid(True)
    plt.legend()
    plt.show()



# ==============================================================
# ЗАДАНИЕ 4. Иллюстрация наложения частот (без построения y_cont)
# ==============================================================
fs_alias = 100   #  неверная частота
N_alias = int(fs_alias * duration)
n_alias = np.arange(N_alias)
t_samples_alias = n_alias / fs_alias
x_samples_alias = np.sin(2 * np.pi * nu1 * t_samples_alias) + np.sin(2 * np.pi * nu2 * t_samples_alias)

# sinc-интерполяция (ряд Котельникова)
def kotelnikov_reconstruct(xn, fs, t_grid):
    n = np.arange(len(xn))
    S = np.sinc(fs * t_grid[:, None] - n[None, :])
    return S @ xn

# Восстановленный сигнал из отсчётов
x_rec_from_alias = kotelnikov_reconstruct(x_samples_alias, fs_alias, t_cont)

# Графики
plt.figure(figsize=(10,5))
plt.plot(t_cont, x_cont, label='Оригинал x(t)')
plt.plot(t_cont, x_rec_from_alias, label='Восстановлен из отсчётов (fs = 100 Гц)')
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

# Загружаем изображение (по твоему пути)
fd = skimage.io.imread(r"C:\Users\vladr\OneDrive\Documents\Учеба МИЭТ\3 курс\ЦОС\var1.png", as_gray=True)
M, N = fd.shape  # размеры изображения

plt.figure()
plt.imshow(fd, cmap='gray')
plt.title("Исходное изображение")
plt.axis("off")
plt.show()

# Пробуем k = 2, 3, 4
for k in [2, 3, 4]:
    # Прореживание изображения
    ff = fd[::k, ::k]
    Mk, Nk = ff.shape

    # Функции Котельникова задаём таблично
    ColumnInd = np.arange(0, max(Mk, Nk))
    SincArray = np.zeros((max(M, N), max(Mk, Nk)))
    for j in range(max(M, N)):
        SincArray[j] = np.sinc(j / k - ColumnInd)

    # Восстановление изображения по формуле Котельникова
    F = SincArray[:M, :Mk] @ ff @ np.transpose(SincArray[:N, :Nk])

    # Отображаем результат
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(ff, cmap='gray')
    plt.title(f"Прореженное изображение (k={k})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(F, cmap='gray')
    plt.title(f"Восстановленное изображение (k={k})")
    plt.axis("off")

    plt.show()
