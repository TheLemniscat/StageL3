import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import special



def plot_gaussian_distribution(mean=0, std_dev=1, highlight_range=(-1, 1)):

    # Generate x values
    x = np.linspace(-5, 5, 1000)
    # Calculate the Gaussian distribution
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.set_title(R'Distribution de $X$', size=25)
    ax.set_xlabel(R'$x$', size=20)
    ax.set_ylabel(R'Densité de probabilité', size=20)
    
    plt.savefig('gaussian_distribution.png', dpi=300)
    plt.show()


def plot_gaussian_CDF(mean=0, std_dev=1):
    # Generate x values
    x = np.linspace(-5, 5, 1000)

    # Calculate the Gaussian CDF
    y = 0.5 * (1 + special.erf((x - mean) / (std_dev * np.sqrt(2))))

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y, color='blue')
    ax.set_title(R'Fonction de répartition de $X$', size=25)
    ax.set_xlabel(R'$x$', size=20)
    ax.set_ylabel(R'$F_X(x)$', size=20)
    plt.savefig('gaussian_cdf.png', dpi=300)
    plt.show()


def plot_heteroscedasticity():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(-3, 3)
    ax.set_title("Hétéroscédasticité", size=25)
    ax.set_xlabel(R'$x$', size=20)
    ax.set_ylabel(R'$y$', size=20)

    x = np.linspace(0, 10, 100)
    line, = ax.plot(x, np.sin(x), color='blue')

    def update(frame):
        noise = np.random.normal(0, 0.1 + frame * 0.05, size=x.shape)
        line.set_ydata(np.sin(x) + noise)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=100, blit=True, interval=100)
    plt.show()



if __name__ == "__main__":
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}"  # facultatif
    })
    plot_heteroscedasticity()