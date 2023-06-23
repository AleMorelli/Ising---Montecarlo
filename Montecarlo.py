#Importar librerías
import numpy
from collections import defaultdict
from matplotlib import pyplot
import itertools

#Definición de constantes(lenght=longitud del sistema,J=constante de intercambio,kB=Constante de Boltzmann)
length = 10
J = 1.0
kB = 1.0

#Creación de los arreglos(sites=almacenamiento de las parejas i,j, spins=diccionario en el que las keys son las parejas
#y los values son los valores de spin, nbhs=defaultdict(lista donde por defecto los values por defecto son listas vacías)
#cada key es una pareja i,j y cada value es una lista de tuplas, donde cada una corresponde a un vecino)
sites = list()
spins = dict()
nbhs = defaultdict(list)

#Creación de la muestra (red cuadrada de spines)
for x, y in itertools.product(range(length), range(length)):
    sites.append((x,y))

#Visualizar sites
print(sites)

#Creación del estado aleatorio(lugar donde se comienzan a estudiar los vecinos)
def random_configuration():
    for spin in sites:
        spins[spin] = numpy.random.choice([-1, 1])

#Configurar y visualizar spines
random_configuration()
print(spins)

#Función de visualización de spines
def plot_spins():
    pyplot.figure()
    colors = {1: "red", -1: "blue"}
    for site, spin in spins.items():
        x, y = site
        pyplot.quiver(x, y, 0, spin, pivot="middle", color=colors[spin])
    pyplot.xticks(range(-1,length+1))
    pyplot.yticks(range(-1,length+1))
    pyplot.gca().set_aspect("equal")
    pyplot.grid()
    pyplot.show
#Ejecución de la función de visualización de spines
plot_spins()


#Asignación de vecinos
nbhs = defaultdict(list)
for site in spins:
    x, y = site
    if x + 1 < length:
        nbhs[site].append(((x + 1) % length, y))
    if x - 1 >= 0:
        nbhs[site].append(((x - 1) % length, y))
    if y + 1 < length:
        nbhs[site].append((x, (y + 1) % length))
    if y - 1 >= 0:
        nbhs[site].append((x, (y - 1) % length))
#Función de cálculo de energía local
def energy_site(site):
    energy = 0.0
    for nbh in nbhs[site]:
        energy += spins[site] * spins[nbh]
    return -J * energy

#Función de cálculo de energía total
def total_energy():
    energy = 0.0
    for site in sites:
        energy += energy_site(site)
    return 0.5 * energy

#Cálculo de la magnetización
def magnetization():
    mag = 0.0
    for spin in spins.values():
        mag += spin
    return mag

#Ejecución de las funciones en el estado 0

plot_spins()
print("magnetization = ", magnetization())

#Implementación del algoritmo "metrópolis"
def metropolis(site, T):
    oldSpin = spins[site]
    oldEnergy = energy_site(site)
    spins[site] *= -1
    newEnergy = energy_site(site)
    deltaE = newEnergy - oldEnergy
    if deltaE <= 0:
        pass
    else:
        if numpy.random.uniform(0, 1) <= numpy.exp(-deltaE/(kB*T)):
            pass
        else:
            spins[site] *= -1
#Implementación del algoritmo "montecarlo"
def monte_carlo_step(T):
    for i in range(len(sites)):
        int_rand_site = numpy.random.randint(0, len(sites))
        rand_site = sites[int_rand_site]
        metropolis(rand_site, T)
#Definir los parámetros para la simulación
amount_mcs = 100000
T_high = 5.0
T_low = 0.01
step = -0.1
#Ciclo de temperatura
#%%time
temps = numpy.arange(T_high, T_low, step)
energies = numpy.zeros(shape=(len(temps), amount_mcs))
magnetizations = numpy.zeros(shape=(len(temps), amount_mcs))
random_configuration()
for ind_T, T in enumerate(temps):
    for i in range(amount_mcs):
        monte_carlo_step(T)
        energies[ind_T, i] = total_energy()
        magnetizations[ind_T, i] = magnetization()

#Observar el estado final a T=0,01
plot_spins()
#Graficos del sistema
#Cálculos de los promedios
tau = amount_mcs // 2
energy_mean = numpy.mean(energies[:, tau:], axis=1)
magnetization_mean = abs(numpy.mean(magnetizations[:, tau:], axis=1))
#Gráficar la energía total y la magnetización en función de la temperatura

fig,axes = pyplot.subplots(2,2,gridspec_kw={'height_ratios':[1,1]})
fig.set_size_inches(13.0,9.0)

ar,ac=0,0
#axes[ar,ac].figure()
axes[ar,ac].plot(temps, energy_mean, label="Energy")
axes[ar,ac].legend()
axes[ar,ac].set_xlabel(r"$T$")
axes[ar,ac].set_ylabel(r"$\left<E\right>$")
axes[ar,ac].grid()
#axes[ar,ac].show()


ar,ac=0,1
#axes[ar,ac].figure()
axes[ar,ac].plot(temps, magnetization_mean, label="Magnetization")
axes[ar,ac].legend()
axes[ar,ac].set_xlabel(r"$T$")
axes[ar,ac].set_ylabel(r"$\left<M\right>$")
axes[ar,ac].grid()
#axes[ar,ac].show()
#Cálculo de la susceptibilidad magnética
magnetization_std = numpy.std(numpy.abs(magnetizations[:, tau:]), axis=1)
susceptibility = magnetization_std ** 2 / (kB * temps)

ar,ac=1,0
#axes[ar,ac].figure()
axes[ar,ac].plot(temps, susceptibility, label="Susceptibility")
axes[ar,ac].legend()
axes[ar,ac].set_xlabel(r"$T$")
axes[ar,ac].set_ylabel(r"$\chi$")
axes[ar,ac].grid()
#axes[ar,ac].show()
#Cállculo del calor específico
energy_std = numpy.std(energies[:, tau:], axis=1)
specific_heat = energy_std ** 2 / (kB * temps * temps)

ar,ac=1,1
#axes[ar,ac].figure()
axes[ar,ac].plot(temps, specific_heat, label="Specific heat")
axes[ar,ac].legend()
axes[ar,ac].set_xlabel(r"$T$")
axes[ar,ac].set_ylabel(r"$C_v$")
axes[ar,ac].grid()
#axes[ar,ac].show()

fig.tight_layout()
pyplot.show()