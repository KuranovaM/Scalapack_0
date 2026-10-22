import struct


def main():
    print("--- Генератор бинарных файлов ---")
    N = int(input("Введите размерность системы N: "))
    mode = int(input("Введите тип (0 - число, 1 - вектор, 2 - матрица): "))
    filename = input("Введите имя файла для сохранения (например, ro.bin): ")

    # Открываем файл на запись в бинарном режиме
    with open(filename, 'wb') as f:

        # --- РЕЖИМ 0: ОДНО ЧИСЛО ---
        if mode == 0:
            val = float(input("Введите 1 число: "))
            # complex<double> занимает 16 байт: 8 на real, 8 на imag
            f.write(struct.pack('<dd', val, 0.0))
            print(f"Готово! Число {val} записано в {filename}")

        # --- РЕЖИМ 1: ВЕКТОР ---
        elif mode == 1:
            print(f"Введите {N} чисел через пробел:")
            nums = list(map(float, input().split()))
            if len(nums) != N:
                print(f"ОШИБКА: вы ввели {len(nums)} чисел, а нужно {N}")
                return

            for val in nums:
                f.write(struct.pack('<dd', val, 0.0))
            print(f"Готово! Вектор записан в {filename}")

        # --- РЕЖИМ 2: МАТРИЦА ---
        elif mode == 2:
            print(f"Введите {N * N} чисел через пробел (вводите по строкам, как обычную матрицу):")
            nums = list(map(float, input().split()))
            if len(nums) != N * N:
                print(f"ОШИБКА: вы ввели {len(nums)} чисел, а нужно {N * N}")
                return

            # Разбиваем введенный список на матрицу NxN (по строкам)
            matrix = []
            for i in range(N):
                row = nums[i * N: (i + 1) * N]
                matrix.append(row)

            # ВАЖНО: ScaLAPACK ждет column-major (по столбцам).
            # Поэтому мы идем по столбцам и берем элементы из нашей матрицы.
            for j in range(N):
                for i in range(N):
                    val = matrix[i][j]
                    f.write(struct.pack('<dd', val, 0.0))

            print(f"Готово! Матрица записана в {filename}")

        else:
            print("ОШИБКА: Неизвестный тип. Введите 0, 1 или 2.")


if __name__ == "__main__":
    main()