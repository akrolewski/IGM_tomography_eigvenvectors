import numpy as np

def compute_deformation_tensor(dm_density):
	delta_k = np.fft.fftn(dm_density)
	kx = np.fft.fftfreq(len(delta_k))
	ky = np.fft.fftfreq(len(delta_k))
	kz = np.fft.fftfreq(len(delta_k))
	kx_3d,ky_3d,kz_3d = np.meshgrid(kx,ky,kz)
	ksq_3d = kx_3d**2 + ky_3d**2 + kz_3d**2

	t_kxkx = kx_3d*kx_3d*delta_k/ksq_3d
	t_kxky = kx_3d*ky_3d*delta_k/ksq_3d
	t_kxkz = kx_3d*kz_3d*delta_k/ksq_3d
	t_kyky = ky_3d*ky_3d*delta_k/ksq_3d
	t_kykz = ky_3d*kz_3d*delta_k/ksq_3d
	t_kzkz = kz_3d*kz_3d*delta_k/ksq_3d

	t_kxkx[0,0,0] = 0
	t_kxky[0,0,0] = 0
	t_kxkz[0,0,0] = 0
	t_kyky[0,0,0] = 0
	t_kykz[0,0,0] = 0
	t_kzkz[0,0,0] = 0

	t_xx = np.fft.ifftn(t_kxkx)
	t_xy = np.fft.ifftn(t_kxky)
	t_xz = np.fft.ifftn(t_kxkz)
	t_yy = np.fft.ifftn(t_kyky)
	t_yz = np.fft.ifftn(t_kykz)
	t_zz = np.fft.ifftn(t_kzkz)

	t_xx = np.real(t_xx)
	t_xy = np.real(t_xy)
	t_xz = np.real(t_xz)
	t_yy = np.real(t_yy)
	t_yz = np.real(t_yz)
	t_zz = np.real(t_zz)

	t_ij = np.array([[t_xx,t_xy,t_xz],[t_xy,t_yy,t_yz],[t_xz,t_yz,t_zz]])
	t_ij_trans = np.transpose(t_ij,axes=(2,3,4,0,1))

	e = np.linalg.eigh(t_ij_trans)
	return e