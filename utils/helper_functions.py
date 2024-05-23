#gernerate smaple tasks
#claculate delay
#calculate waiting time
# ...
import numpy as np 
import matplotlib.pyplot as plt
 
def generate_samlpe_tasks(n , m):
    tasks = [(i, np.randon.randint(1,50) for x in range(m)) for i in range(20)]    

    return tasks


def plot_chart(self, x_values, y_values, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()