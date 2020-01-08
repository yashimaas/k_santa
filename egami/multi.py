import time
import concurrent.futures


# 単に時間がかかるだけの処理
def killing_time(number):
    return_list = []
    for i in range(1, number + 1):
        if number % i == 1:
            if i <= 9999:
                return_list.append(i)
    return return_list


def main():
    start = time.time()
    num_list = [25000000, 20000000, 20076000, 14500000]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as excuter:
        result_list = list(excuter.map(killing_time, num_list))
    stop = time.time()
    print('%.3f seconds' % (stop - start))


if __name__ == '__main__':
    main()
    