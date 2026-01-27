import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
from matplotlib.animation import FuncAnimation

class GaussianBeam():
    def __init__(self, N, l, size_screen, w0, x_0 = 0, y_0 = 0):
        x = np.linspace(-l/2, l/2, N)
        y = np.linspace(-l/2, l/2, N)
        self.xx, self.yy = np.meshgrid(x, y)
        self.N = N
        self.l = l
        self.size_screen = size_screen
        self.w0 = w0
        self.E = np.exp(-((self.xx - x_0)**2 + (self.yy-y_0)**2)/w0**2)

    def get_z(self):
        return self.z
    
    def get_xy(self):
        return [self.xx, self.yy]
    
    def get_circular_aperture(self, D):
        mask = Propagator.get_mask(self.N, self.size_screen, 0.8*self.size_screen)
        print(mask)
        return self.E * mask
    
    def show_gaussin_beam(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.xx, self.yy, self.z)
        ax.set_xlabel('X ось')
        ax.set_ylabel('Y ось')
        ax.set_zlabel('Z ось')  
        plt.show()
    
    def show_intensivity(self, R = None):
        if R is None:
            R = self.l
        E = self.get_circular_aperture(R)
        intensivity = np.abs(E)**2
        fig = plt.figure(figsize=(16, 9))
        im1 = plt.imshow(intensivity, extent=[-R/2, R/2, -R/2, R/2], cmap='hot')
        plt.colorbar(im1)

    def show_phase(self, R = None):
        if R is None:
            R = self.l
        phase = np.angle(self.E)*Propagator.get_mask(self.N, self.size_screen, 0.8*self.size_screen)
        fig = plt.figure(figsize=(16, 9))
        im1 = plt.imshow(phase, extent=[-self.l/2, self.l/2, -self.l/2, self.l/2], cmap='viridis')
        plt.colorbar(im1)

class ApertureFunctions:

    ZERNIKE_FUNCTIONS = {
        0: lambda x, y: np.ones_like(x),
        1: lambda x, y: x,
        2: lambda x, y: y,
        3: lambda x, y: 2*(x**2 + y**2) - 1,
        4: lambda x, y: x**2 - y**2,
        5: lambda x, y: 2*x*y,
        6: lambda x, y: (3*(x**2 + y**2) - 2) * x,
        7: lambda x, y: (3*(x**2 + y**2) - 2) * y,
        8: lambda x, y: 6*(x**2 + y**2)**2 - 6*(x**2 + y**2) + 1,
        9: lambda x, y: x**3 - 3*x*y**2,
        10: lambda x, y: 3*x**2*y - y**3,
        11: lambda x, y: (4*(x**2 + y**2) - 3) * (x**2 - y**2),
        12: lambda x, y: (4*(x**2 + y**2) - 3) * (2*x*y),
        13: lambda x, y: (10*(x**2 + y**2)**2 - 12*(x**2 + y**2) + 3) * x,
        14: lambda x, y: (10*(x**2 + y**2)**2 - 12*(x**2 + y**2) + 3) * y,
        15: lambda x, y: 20*(x**2 + y**2)**3 - 30*(x**2 + y**2)**2 + 12*(x**2 + y**2) - 1,
        16: lambda x, y: (5*(x**2 + y**2) - 4) * (x**3 - 3*x*y**2),
        17: lambda x, y: (5*(x**2 + y**2) - 4) * (3*x**2*y - y**3),
        18: lambda x, y: 15*(x**2 + y**2)**4 - 20*(x**2 + y**2)**3 + 12*(x**2 + y**2)**2 - 1,
        19: lambda x, y: x**4 - 6*x**2*y**2 + y**4,
        20: lambda x, y: 4*x*y * (x**2 - y**2),
        21: lambda x, y: (15*(x**2 + y**2)**2 - 20*(x**2 + y**2) + 6) * (x**2 - y**2),
        22: lambda x, y: (15*(x**2 + y**2)**2 - 20*(x**2 + y**2) + 6) * (2*x*y),
        23: lambda x, y: (35*(x**2 + y**2)**3 - 60*(x**2 + y**2)**2 + 30*(x**2 + y**2) - 4) * x,
        24: lambda x, y: (35*(x**2 + y**2)**3 - 60*(x**2 + y**2)**2 + 30*(x**2 + y**2) - 4) * y,
        25: lambda x, y: 70*(x**2 + y**2)**4 - 140*(x**2 + y**2)**3 + 90*(x**2 + y**2)**2 - 20*(x**2 + y**2) + 1,
        26: lambda x, y: x**5 - 10*x**3*y**2 + 5*x*y**4,
        27: lambda x, y: 5*x**4*y - 10*x**2*y**3 + y**5,
        28: lambda x, y: (6*(x**2 + y**2) - 5) * (x**5 - 10*x**3*y**2 + 5*x*y**4),
        29: lambda x, y: (6*(x**2 + y**2) - 5) * (5*x**4*y - 10*x**2*y**3 + y**5),
        30: lambda x, y: (21*(x**2 + y**2)**2 - 30*(x**2 + y**2) + 10) * (x**3 - 3*x*y**2),
        31: lambda x, y: (21*(x**2 + y**2)**2 - 30*(x**2 + y**2) + 10) * (3*x**2*y - y**3),
        32: lambda x, y: (56*(x**2 + y**2)**3 - 105*(x**2 + y**2)**2 + 60*(x**2 + y**2) - 10) * (x**2 - y**2),
        33: lambda x, y: (56*(x**2 + y**2)**3 - 105*(x**2 + y**2)**2 + 60*(x**2 + y**2) - 10) * (2*x*y),
        34: lambda x, y: (126*(x**2 + y**2)**4 - 280*(x**2 + y**2)**3 + 210*(x**2 + y**2)**2 - 60*(x**2 + y**2) + 5) * x,
        35: lambda x, y: (126*(x**2 + y**2)**4 - 280*(x**2 + y**2)**3 + 210*(x**2 + y**2)**2 - 60*(x**2 + y**2) + 5) * y,
        36: lambda x, y: 252*(x**2 + y**2)**5 - 630*(x**2 + y**2)**4 + 560*(x**2 + y**2)**3 - 210*(x**2 + y**2)**2 + 30*(x**2 + y**2) - 1,
        37: lambda x, y: x**6 - 15*x**4*y**2 + 15*x**2*y**4 - y**6,
        38: lambda x, y: 6*x**5*y - 20*x**3*y**3 + 6*x*y**5,
        39: lambda x, y: (7*(x**2 + y**2) - 6) * (x**6 - 15*x**4*y**2 + 15*x**2*y**4 - y**6),
        40: lambda x, y: (7*(x**2 + y**2) - 6) * (6*x**5*y - 20*x**3*y**3 + 6*x*y**5),   
    }

    @classmethod
    def zernike_j(cls, j, X, Y):
        return cls.ZERNIKE_FUNCTIONS[j](X, Y)

    @classmethod
    def zernike(cls, coeffs, X, Y):
        res = 0
        for ind in range(len(coeffs)):
            res += coeffs[ind]*cls.zernike_j(ind, X, Y)

        R = np.sqrt(X**2 + Y**2)
        mask = R <= 1.0

        return res*mask
    
    @classmethod
    def zernike_image(cls, coeffs, N):
        X = np.linspace(-1, 1, N)
        Y = X.copy()
        X, Y = np.meshgrid(X, Y)

        return cls.zernike(coeffs, X, Y)
    
    @classmethod
    def reconstruct_zernike(phase_measured, X_norm, Y_norm, num_coeffs=15):
        R = np.sqrt(X_norm**2 + Y_norm**2)
        aperture_mask = R <= 1
        
        coeffs = []
        for j in range(num_coeffs):
            Z_j = ApertureFunctions.zernike_j(j, X_norm, Y_norm)
            
            numerator = np.sum(phase_measured * Z_j * aperture_mask)
            denominator = np.sum(Z_j**2 * aperture_mask)
            
            coeff = numerator / denominator if denominator != 0 else 0
            coeffs.append(coeff)
        
        return np.array(coeffs)

class Propagator():
    @classmethod
    def get_phase_screen(cls, N, dx, r0, L0=1.0, l0=0.001):

        kx = 2 * np.pi * np.fft.fftfreq(N, dx)
        ky = 2 * np.pi * np.fft.fftfreq(N, dx)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        

        P_phi = (0.023 * r0**(-5/3) * np.exp(-(K**2) * (l0**2)/(2*np.pi)**2) / ((K**2 + (2*np.pi/L0)**2)**(11/6)))
    
        P_phi = np.where(K > 1e-6, P_phi, 0)

        rng = np.random.default_rng()
        phase_fft = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * np.sqrt(P_phi)
        phase_screen = np.real(ifft2(phase_fft)) * (N / dx)**2
        return phase_screen / np.std(phase_screen)
    
    @classmethod
    def propagate(cls, N, dx, E0, wavelength, distance, method='exact'):

        E0_fft = fft2(E0)
        E0_fft_shifted = fftshift(E0_fft)

        fx = fftfreq(N, dx) 
        fy = fftfreq(N, dx) 
        FX, FY = np.meshgrid(fx, fy, indexing='xy')
        FX = fftshift(FX)
        FY = fftshift(FY)

        k = 2 * np.pi / wavelength 

        if method == 'paraxial':
            H = np.exp(1j * k * distance) * np.exp(-1j * np.pi * wavelength * distance * (FX ** 2 + FY ** 2))
        elif method == 'exact':
            kx = 2 * np.pi * FX
            ky = 2 * np.pi * FY
            kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2 + 0j)
            H = np.exp(1j * kz * distance)
            H[kx ** 2 + ky ** 2 > k ** 2] = 0

        Ez_fft_shifted = E0_fft_shifted * H

        Ez_fft = ifftshift(Ez_fft_shifted)
        E = ifft2(Ez_fft)
        return E
    
class ShackHartmann:
    def __init__(self, size: int = 1024, num_subapertures: int = 4, period: float = 0.011, 
                 radius_subapertures: float = 0.005, focal_length: float = 0.1, 
                 wavelength: float = 1e-6, mode = 'square'):
        
        self.size = size
        self.num_subapertures = num_subapertures
        self.period = period
        self.radius_subapertures = radius_subapertures
        self.focal_length = focal_length
        self.wavelength = wavelength
        self.mode = mode
        
        self.pixel_size = (num_subapertures * period) / size
        self.k = 2 * np.pi / wavelength
        self.size_one_subaperures = int(self.size / self.num_subapertures)
        self.X, self.Y = self._create_coordinate_grid()
        self.coordinate_centers = self._get_subaperture_centers()
        
        self.diameter_px = int(np.ceil(2 * self.radius_subapertures / self.pixel_size))
        if self.diameter_px % 2 == 0:
            self.diameter_px += 1
        self.half_diameter = self.diameter_px // 2
        
        self.aperture_mask = self.create_aperture_mask()
        self.phase_mask = self.create_phase_mask()
        
    
    def _create_coordinate_grid(self):
        x = np.linspace(-self.size/2, self.size/2, self.size) * self.pixel_size
        y = np.linspace(-self.size/2, self.size/2, self.size) * self.pixel_size
        return np.meshgrid(x, y)
    
    def _get_subaperture_centers(self):
        centers = []
        n = self.num_subapertures
        indices = (np.arange(n) - (n-1)/2) * 1.0
        
        for i in indices:
            for j in indices:
                center_x = i * self.period
                center_y = j * self.period
                
                if self.mode == 'square':
                    centers.append((center_x, center_y))
                elif self.mode == 'radial':
                    distance = np.sqrt(center_x**2 + center_y**2)
                    max_radius = (n * self.period) / 2
                    if distance <= max_radius:
                        centers.append((center_x, center_y))
        
        return np.array(centers)
    
    def create_aperture_mask(self):
        aperture_mask = np.zeros((self.size, self.size), dtype=np.float32)
        one_aperture_mask = np.zeros((self.diameter_px, self.diameter_px), dtype=np.float32)
        
        x = np.linspace(-self.half_diameter, self.half_diameter, self.diameter_px) * self.pixel_size
        y = np.linspace(-self.half_diameter, self.half_diameter, self.diameter_px) * self.pixel_size
        x, y = np.meshgrid(x, y)
        
        r_squared = x**2 + y**2
        one_aperture_mask[r_squared <= self.radius_subapertures**2] = 1
        
        center_offset = self.size // 2
        
        for center_x, center_y in self.coordinate_centers:
            center_x_px = int(round(center_x / self.pixel_size)) + center_offset
            center_y_px = int(round(center_y / self.pixel_size)) + center_offset
            
            x_start = center_x_px - self.half_diameter
            x_end = x_start + self.diameter_px
            y_start = center_y_px - self.half_diameter
            y_end = y_start + self.diameter_px
            
            if x_start < 0:
                temp_x_start = -x_start
                x_start = 0
            else:
                temp_x_start = 0
            if x_end > self.size:
                temp_x_end = self.diameter_px - (x_end - self.size)
                x_end = self.size
            else:
                temp_x_end = self.diameter_px
            
            if y_start < 0:
                temp_y_start = -y_start
                y_start = 0
            else:
                temp_y_start = 0
            if y_end > self.size:
                temp_y_end = self.diameter_px - (y_end - self.size)
                y_end = self.size
            else:
                temp_y_end = self.diameter_px
            
            aperture_mask[y_start:y_end, x_start:x_end] = one_aperture_mask[temp_y_start:temp_y_end, temp_x_start:temp_x_end]
        
        return aperture_mask
    
    def create_phase_mask(self):
        phase_mask = np.zeros((self.size, self.size), dtype=np.float32)
        
        lens_template = np.zeros((self.diameter_px, self.diameter_px), dtype=np.float32)
        x = np.linspace(-self.half_diameter, self.half_diameter, self.diameter_px) * self.pixel_size
        y = np.linspace(-self.half_diameter, self.half_diameter, self.diameter_px) * self.pixel_size
        X_temp, Y_temp = np.meshgrid(x, y)
        R_sq_temp = X_temp**2 + Y_temp**2
        
        lens_phase = -R_sq_temp * self.k / (2 * self.focal_length)
        lens_mask = R_sq_temp <= self.radius_subapertures**2
        lens_template[lens_mask] = lens_phase[lens_mask]
        
        center_offset = self.size // 2
        
        for center_x, center_y in self.coordinate_centers:
            center_x_px = int(round(center_x / self.pixel_size)) + center_offset
            center_y_px = int(round(center_y / self.pixel_size)) + center_offset
            
            x_start = center_x_px - self.half_diameter
            x_end = x_start + self.diameter_px
            y_start = center_y_px - self.half_diameter
            y_end = y_start + self.diameter_px
            
            if x_start < 0:
                temp_x_start = -x_start
                x_start = 0
            else:
                temp_x_start = 0
            if x_end > self.size:
                temp_x_end = self.diameter_px - (x_end - self.size)
                x_end = self.size
            else:
                temp_x_end = self.diameter_px
            
            if y_start < 0:
                temp_y_start = -y_start
                y_start = 0
            else:
                temp_y_start = 0
            if y_end > self.size:
                temp_y_end = self.diameter_px - (y_end - self.size)
                y_end = self.size
            else:
                temp_y_end = self.diameter_px
            
            phase_mask[y_start:y_end, x_start:x_end] += lens_template[temp_y_start:temp_y_end, temp_x_start:temp_x_end]
        
        return phase_mask

    def E_focal(self, E, method = 'exact'):
        E_0 = E * self.aperture_mask * np.exp(1j * self.phase_mask)
        E_total = Propagator.propagate(E_0.shape[0], self.pixel_size, E_0, self.wavelength, self.focal_length, method)
        return E_total