from AO import ShackHartmann, GaussianBeam, ApertureFunctions, Propagator
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

N = 1024
dx = 1e-4
L_screen = N*dx
wavelength = 650e-9
phase = np.zeros([N, N])

gauss_0 = GaussianBeam(N, L_screen, L_screen, L_screen/30, x_0=L_screen/5)
gauss_1 = GaussianBeam(N, L_screen, L_screen, L_screen/30, x_0=-L_screen/5)
E = 2*gauss_0.E*np.exp(1j*phase) + 2*gauss_1.E*np.exp(1j*phase)

prop = Propagator()

n_phase_screen = 80
dist = 5

E_all = [np.abs(E)**2]
phase_all = [phase]

for i in range(n_phase_screen):
    E = Propagator.propagate(N, dx, E, wavelength, dist, method = 'exact')

    intensivity_i = np.abs(E)**2
    phase_i = np.angle(E)

    print(i, intensivity_i.mean())

    E_all.append(intensivity_i)
    phase_all.append(phase_i)

l = L_screen*1000

fig = plt.figure(figsize=(11, 11))

plt.subplot(2, 2, 1)
plt.imshow(E_all[0], cmap = 'hot', extent=[-l/2, l/2, -l/2, l/2])
plt.title(f'L = {0} м')
plt.xlabel('x, мм')
plt.ylabel('y, мм')
plt.clim(0, 1)
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(E_all[n_phase_screen//4], cmap = 'hot', extent=[-l/2, l/2, -l/2, l/2])
plt.title(f'L = {dist*n_phase_screen//4} м')
plt.xlabel('x, мм')
plt.ylabel('y, мм')
plt.clim(0, 1)
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(E_all[n_phase_screen//2], cmap = 'hot', extent=[-l/2, l/2, -l/2, l/2])
plt.title(f'L = {dist*n_phase_screen//2} м')
plt.xlabel('x, мм')
plt.ylabel('y, мм')
plt.clim(0, 1)
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(E_all[n_phase_screen], cmap = 'hot', extent=[-l/2, l/2, -l/2, l/2])
plt.title(f'L = {dist*n_phase_screen} м')
plt.colorbar()
plt.xlabel('x, мм')
plt.ylabel('y, мм')
plt.clim(0, 1)
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(np.zeros((N, N)), cmap='hot', extent=[-l/2, l/2, -l/2, l/2])

plt.colorbar(im, label='Интенсивность')
ax.set_title('Интенсивность')

def init():
    im.set_data(np.zeros((N, N)))
    im.set_clim(0, 1)
    return im,

def update(frame):
    im.set_data(E_all[frame])
    ax.set_title(f'Интенсивность, пройденное расстояние L = {frame*dist} м')
    return im,

ani = FuncAnimation(fig, update, frames=n_phase_screen+1,
                    init_func=init,
                    interval=30,
                    blit=True,
                    repeat=True)

ani.save('animation_np.gif', writer='pillow', fps=15, dpi=100)
plt.tight_layout()
plt.show()
