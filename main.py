import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import tkinter as tk
import os
from tkinter import PhotoImage
from PIL import Image, ImageTk
class Network_Work:
    #Конструктор класса
    def __init__(self, n):
        self.graph = nx.barabasi_albert_graph(n, 2) # Создает граф с использованием модели Барабаши-Альберта с n узлами и степенью добавления 2.
        self.n = n
        # Массив случайных значений от 0 до 0.5, представляющих параметр beta для каждого узла
        self.beta = np.random.rand(n) * 0.5
        # self.matrix_kef: Матрица случайных значений от 0 до 0.2, которая затем обнуляется для несвязанных узлов.
        self.matrix_kef = np.random.rand(n, n) * 0.2
        self.matrix_kef  = np.where(nx.to_numpy_array(self.graph) > 0, self.matrix_kef, 0)
        # self.phi: Массив случайных значений от 0 до 0.1, представляющих параметр phi для каждого узла.
        self.phi = np.random.rand(n) * 0.1
    # Методы графа:
    # Возвращает граф
    def get_graph(self):
        return self.graph
    # Возвращает массив beta
    def get_beta(self):
        return self.beta
    # Возвращает матрицу коэффициентов.
    def get_matrix_kef(self):
        return self.matrix_kef
    # Возвращает массив phi.
    def get_phi(self):
        return self.phi

class Epidemic_Model:
    # Конструктор класса
    def __init__(self, network, T, initial_node,  steps):
        self.network = network # Ассоциация с классом  Network_Work
        self.T = T
        self.initial_node = initial_node
        self.steps = steps
        self.res = None

    def simulate_epidemic(self):
        graph = self.network.get_graph()
        beta = self.network.get_beta()
        matrix_kef = self.network.get_matrix_kef()
        matrixkeff = self.network.get_phi()

        n = graph.number_of_nodes()
        res = {node: [0] * self.steps for node in graph.nodes()}
        res[self.initial_node][0] = 1
        number = 1
        nn = 1
        while True:
            docX = f"{nn}.text"
            # Проверяем, существует ли файл
            if not os.path.exists(docX):
                print(f"Файл '{docX}' не найден. Продолжаем поиск...")
                break  # Выходим из цикла, если файл найден
            else:
                print(f"Файл '{docX}' найден!")
                nn += 1  # Переходим к следующему номеру
        with open(docX, "a", encoding="utf-8") as file:
            file.write("Матрица коэффициентов:")
            file.write("\n")
            for row in matrix_kef:
                file.write(" ".join(map(str, row)) + "\n")
            file.write("\n")
        for step in range(1, self.steps):
            new_res = res.copy()
            for node in graph.nodes():
                if res[node][step-1] > 0:
                    new_res[node][step] = res[node][step-1] + (beta[node] * self.T)
                for neighbor in graph.neighbors(node):
                    transfer = matrix_kef[node][neighbor] * res[node][step-1]
                    new_res[neighbor][step] = max(0, res[neighbor][step-1] + transfer + (beta[neighbor] - matrixkeff[neighbor]) * self.T)
            res = new_res

            # Вывод результатов на каждом шаге в консоль
            print(f"Шаг {step}:")
            b = f"Шаг {step}:"
            with open(docX, "a", encoding="utf-8") as file:
                file.write(b.replace("\n", " "))
                file.write("\n")
                file.write(b.replace("\n", " "))
                file.write("\n")
            for node in graph.nodes():
                print(f"Узел {node + 1}: {res[node][step]}")
                c = f"Узел {node + 1}: {res[node][step]}"
                with open(docX, "a", encoding="utf-8") as file:
                    file.write(c.replace("\n", " "))
                with open(docX, "a", encoding="utf-8") as file:
                    file.write("\n")

        for node in graph.nodes():
            graph.nodes[node]['res'] = res[node][-1]
        self.res = res
        return res, graph


class Epidemic_Visualizer:
    # Конструктор класса
    def __init__(self, network, res, steps):
        self.network = network
        self.res = res
        self.steps = steps
    # График результатов. Строятся графики
    def plot_results(self):
        graph = self.network.get_graph()
        plt.figure(figsize=(10, 5))
        plt.suptitle("Моделирование прогнозируемой эпидемии", fontsize=16)
        node_colors = [graph.nodes[node]['res'] for node in graph.nodes()]
        for node, levels in self.res.items():
            plt.plot(levels, label=f"Node {node + 1}", color=plt.cm.plasma(node_colors[node] / max(node_colors)))
        plt.xlabel("Временной промежуток")
        plt.ylabel("Прогнозируемый уровень заражения вирусом")
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 5))
        pos = nx.spring_layout(graph)
        labels = {node: str(node + 1) for node in graph.nodes()}
        nodes = nx.draw_networkx_nodes(graph, pos, node_color=node_colors, cmap=plt.cm.plasma)
        nx.draw_networkx_edges(graph, pos)
        nx.draw_networkx_labels(graph, pos, labels=labels)
        plt.title("График сети")
        plt.colorbar(nodes, label="Уровень заражения вирусом")

        plt.show()
    # Принцип предсказания будущих распространений вируса
    def future(self):
        x = np.array(list(range(self.steps))).reshape(-1, 1)
        y = np.array([self.res[node][-1] for node in self.network.get_graph().nodes()]).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, y)
        future_steps = 2
        future_X = np.array(list(range(self.steps, self.steps + future_steps))).reshape(-1, 1)
        predicted_y = model.predict(future_X)
        plt.figure(figsize=(10, 5))
        for node in self.network.get_graph().nodes():
            # Используем predicted_y[node, 0] для двумерного массива
            plt.plot(range(self.steps, self.steps + future_steps), predicted_y[node, 0],
                     label=f"Предсказанный узел {node + 1}", linestyle='--')


def start_process():
    try:
        n = int(entry1.get())
        t = float(entry2.get())
        initial_node1 = int(entry3.get())
        initial_node = initial_node1 - 1
        steps = int(entry4.get())
        network = Network_Work(n) # Создание объекта network класса Network_Work
        # Затем этот объект передается в конструкторы классов Epidemic_Model и Epidemic_Visualizer
        epidemic_model = Epidemic_Model(network, t, initial_node, steps)
        res, graph = epidemic_model.simulate_epidemic()
        visualizer = Epidemic_Visualizer(network, res, epidemic_model.steps)
        visualizer.plot_results()
        visualizer.future()
    except ValueError:
        text = tk.Label(window1, text="Ошибка ! Введите числа")
        text.grid(row=6, column=1)


# ГРАФИЧЕСКОЕ ОКНО ДЕЛАЕМ:
window1 = tk.Tk()
window1.title("Эпидемическая модели Бахчисарайский фонтан")
window1.geometry("450x300")
window1.attributes("-toolwindow", True)
window1.configure(bg="ghostwhite")
label1 = tk.Label(window1, text="Введите количество узлов ", background="ghostwhite", foreground="navy")
label1.grid(row=0, column=0)
entry1 = tk.Entry(window1)
entry1.grid(row=0, column=1)
label2 = tk.Label(window1, text="Время распространения вируса ", background="ghostwhite", foreground="navy")
label2.grid(row=1, column=0)
entry2 = tk.Entry(window1)
entry2.grid(row=1, column=1)
label3 = tk.Label(window1, text="Какой узел начинает заражать ", background="ghostwhite", foreground="navy")
label3.grid(row=2, column=0)
entry3 = tk.Entry(window1)
entry3.grid(row=2, column=1)
label4 = tk.Label(window1, text="Количество шагов моделирования эпидемии ", background="ghostwhite", foreground="navy")
label4.grid(row=3, column=0)
entry4 = tk.Entry(window1)
entry4.grid(row=3, column=1)
start_button = tk.Button(window1, text="Старт", command=start_process)
start_button.grid(row=5, column=1, columnspan=5)
i = Image.open("1.jpg")  # Укажите путь к вашему изображению
photo = ImageTk.PhotoImage(i)
im = tk.Label(window1, image=photo)
im.grid()
window1.photo = photo
window1.mainloop()