from AO import ShackHartmann, GaussianBeam, ApertureFunctions

import numpy as np
import matplotlib.pyplot as plt

N = 4096
dx = 1e-4
wavelength = 650e-9
phase_0 = ApertureFunctions.zernike_image(np.random.uniform(0, 1, 6), N)
gauss = GaussianBeam(N, N*dx, N*dx, N*dx/2)
E = gauss.E*np.exp(1j*phase_0)

sensor = ShackHartmann(
    size=N, 
    num_subapertures=15,
    period=300e-6,
    radius_subapertures = 136e-6,
    focal_length=0.00322, 
    wavelength=650e-9,
    mode='radial'
)

E_focal = sensor.E_focal(E*sensor.aperture_mask)
intensity_focal = np.abs(E_focal)**2

I_max = max([np.max(np.abs(E)**2), np.max(np.abs(E_focal)**2)])
I_min = min([np.min(np.abs(E)**2), np.min(np.abs(E_focal)**2)])

plt.suptitle('Симуляция работы датчика Шака-Гартмана на примере гауссова пучка со случайным распределением фазы', fontweight='bold')

plt.subplot(2, 2, 1)
plt.imshow(phase_0/(2*np.pi), cmap='hsv')
plt.title('Начальная фаза в долях пи')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(np.abs(E)**2, cmap='hot')
plt.title('Начальная интенсивность')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(sensor.phase_mask)
plt.title('Фазовая маска датчика')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(np.abs(E_focal)**2, cmap = 'hot')
plt.title('Интенсивность в фокальной плоскости')
plt.clim(I_min, I_max)
plt.colorbar()
plt.show()


