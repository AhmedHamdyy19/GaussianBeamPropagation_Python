import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
#           Variables
# -----------------------------------
Wo           = 30e-6
lambda_o     = 2.4e-6
lambda_glass = lambda_o/np.sqrt(2.25)
k1           = 2*np.pi/lambda_o
k2           = 2*np.pi/lambda_glass
eta1         = 377                         # Wave impedance of air
eta2         = 377/np.sqrt(2.25);          # Wave impedance of glass
dx           = 2e-6
zo           = np.pi*(Wo**2)/lambda_o      # Rayleigh distance in air
zo2          = np.pi*(Wo**2)/lambda_glass  # Rayleigh distance in glass
N            = 3000
# -----------------------------------
#         Gaussian Beam
# -----------------------------------
X = 100*Wo
x = np.arange(-X, X, dx)
u = np.exp(-(x**2)/Wo**2)            # Gaussian beam at z=0

plt.figure(1)
plt.plot(x, u**2)
plt.xlabel('x(um)')
plt.ylabel('I/I\u2092')
plt.axis([-2e-4, 2e-4, 0, 1])
plt.grid()
ticks = [-200, -150, -100, -50, 0, 50, 100, 150, 200]
plt.xticks([x/10**6 for x in ticks], ['{}'.format(x) for x in ticks])
plt.title('Gaussian Beam at z = 0')
plt.savefig('atZero.png')
plt.show(block=False)

# -----------------------------------
#        Spatial Transform
# -----------------------------------

Kx = np.arange(-N/2, N/2) * 1/X
Kx = Kx * np.pi
U  = np.fft.fftshift(np.fft.fft(u))           # Fourier transform of Gaussian beam at z = 0

# -----------------------------------
#          Incident Beam
# -----------------------------------

Kz = np.sqrt(k1**2 - Kx**2)
tmp_z = [1, 2]

# Fourier transform of Gaussian beam at different propagation distances
U_z = np.zeros((len(tmp_z), N), dtype=complex)

# Gaussian beam at different propagation distances
u_z = np.zeros((len(tmp_z), N), dtype=complex)

for j in range(len(tmp_z)):
    U_z[j, :] = U * np.exp(-1j * Kz * tmp_z[j] * zo)
    u_z[j, :] = np.abs(np.fft.ifft(np.fft.fftshift(U_z[j, :])))
    plt.figure()
    plt.plot(x, np.abs(u_z[j, :]**2))
    plt.grid()
    plt.xlabel('x(um)')
    plt.ylabel('I/I\u2092')
    plt.xticks([x / 10 ** 6 for x in ticks], ['{}'.format(x) for x in ticks])
    plt.axis([-2e-4, 2e-4, 0, 1])
    txt = 'Gaussian Beam at Z = %gz\u2092' % tmp_z[j]
    plt.title(txt)
    filename = 'inc_%d.png' % j
    plt.savefig(filename)
    plt.show(block=False)

# -----------------------------------
#       Plane Waves Parameters
# -----------------------------------

theta_i = np.arctan(Kx / Kz)
theta_t = np.arcsin(k1 * np.sin(theta_i) / k2)
gamma = (eta2 * np.cos(theta_i) - eta1 * np.cos(theta_t)) \
        / (eta2 * np.cos(theta_i) + eta1 * np.cos(theta_t))
T = (2 * eta2 * np.cos(theta_i)) / (eta2 * np.cos(theta_i) + eta1 * np.cos(theta_t))

# -----------------------------------
#         Transmitted Beam
# -----------------------------------

kz_glass = k2 * np.cos(theta_t)

# Fourier transform of transmitted beam through glass
Uz_transmitted = np.zeros((2, N), dtype=complex)

# Transmitted beam through glass
uz_transmitted = np.zeros((2, N), dtype=complex)


for j in range(len(tmp_z)):
    Uz_transmitted[j, :] = T * U_z[1,:] * np.exp(-1j * kz_glass * tmp_z[j] * zo2)
    uz_transmitted[j, :] = np.abs(np.fft.ifft(Uz_transmitted[j, :]))
    plt.figure()
    plt.plot(x, np.abs(uz_transmitted[j, :]**2))
    plt.grid()
    plt.xlabel('x(um)')
    plt.ylabel('I/I\u2092')
    plt.xticks([x / 10 ** 6 for x in ticks], ['{}'.format(x) for x in ticks])
    plt.axis([-2e-4, 2e-4, 0, 1])
    txt = 'Transmitted Gaussian Beam at Z = %gz\u2092' % tmp_z[j]
    plt.title(txt)
    filename = 'trans_%d.png' % j
    plt.savefig(filename)
    plt.show(block=False)


# -----------------------------------
#          Reflected Beam
# -----------------------------------

plt.figure()
Uz_reflected = gamma * U_z[1,:] * np.exp(-1j * Kz * zo)         # Fourier transform of Reflected beam
uz_reflected = np.abs(np.fft.ifft(Uz_reflected))                # Reflected beam
plt.plot(x, np.abs(uz_reflected**2))
plt.grid()
plt.xlabel('x(um)')
plt.ylabel('I/I\u2092')
plt.xticks([x / 10 ** 6 for x in ticks], ['{}'.format(x) for x in ticks])
plt.axis([-2e-4, 2e-4, 0, 0.1])
txt = 'Reflected Gaussian Beam at Z = z\u2092'
plt.title(txt)
plt.savefig('ref.png')
plt.show()
