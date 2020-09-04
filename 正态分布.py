import numpy as np
import matplotlib.pyplot as plt


def random_event(count=10000):
    dataset = {}
    data = np.random.randint(1, 7, count)
    for i in data:
        if i not in dataset:
            dataset[i] = 1
        dataset[i] += 1
    # print(dataset)
    # print(data.mean()) # 打印平均值
    # print(data.std())  # 打印标准差
    return dataset


def _show(x, y):
    plt.bar(x=x, height=y)
    # 柱状图显示值
    for x, y in zip(x, y):
        plt.text(x, y, y, ha='center', va='bottom')
    plt.show()


def sample_one(count=10000):
    data = np.random.randint(1, 7, count)
    xs = []
    for i in range(10):
        xs.append(data[int(np.random.random() * len(data))])
    npdata = np.array(xs)
    print(npdata.mean())
    print(npdata.std())
    return npdata.mean()


def start():
    # dataset = random_event()
    # x = list(dataset.keys())
    # y = list(dataset.values())
    # _show(x, y)
    data = np.random.randint(1, 10, 100000)
    data_means = []
    datas = {}

    for i in range(10000):
        sample = []
        for j in range(500):
            sample.append(data[int(np.random.random() * len(data))])
        npdata = np.array(sample)
        m = npdata.mean()
        if m not in datas:
            datas[m] = 1
        datas[m] += 1
    sorted_data = sorted(datas.items(), key=lambda x: x[0])
    x = [a[0] for a in sorted_data]
    y = [b[1] for b in sorted_data]

    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    start()
    # sample_one()
