
import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d

class calibration(object):
	'''
	some useful tools for manual calibration
	'''
	def normalize_zdata(self,z_data,cal_z_data):
		return z_data/cal_z_data
		
	def normalize_amplitude(self,z_data,cal_ampdata):
		return z_data/cal_ampdata
		
	def normalize_phase(self,z_data,cal_phase):
		return z_data*np.exp(-1j*cal_phase)
		
	def normalize_by_func(self,f_data,z_data,func):
		return z_data/func(f_data)
		
	def _baseline_als(self,y, lam, p, niter=10):
		'''
		see http://zanran_storage.s3.amazonaws.com/www.science.uva.nl/ContentPages/443199618.pdf
		"Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005.
		http://stackoverflow.com/questions/29156532/python-baseline-correction-library
		"There are two parameters: p for asymmetry and lambda for smoothness. Both have to be
		tuned to the data at hand. We found that generally 0.001<=p<=0.1 is a good choice
		(for a signal with positive peaks) and 10e2<=lambda<=10e9, but exceptions may occur."
		'''
		L = len(y)
		D = sparse.csc_matrix(np.diff(np.eye(L), 2))
		w = np.ones(L)
		for i in range(niter):
			W = sparse.spdiags(w, 0, L, L)
			Z = W + lam * D.dot(D.transpose())
			z = sparse.linalg.spsolve(Z, w*y)
			w = p * (y > z) + (1-p) * (y < z)
		return z
		
	def fit_baseline_amp(self,z_data,lam,p,niter=10):
		'''
		for this to work, you need to analyze a large part of the baseline
		tune lam and p until you get the desired result
		'''
		return self._baseline_als(np.absolute(z_data),lam,p,niter=niter)
	
	def baseline_func_amp(self,z_data,f_data,lam,p,niter=10):
		'''
		for this to work, you need to analyze a large part of the baseline
		tune lam and p until you get the desired result
		returns the baseline as a function
		the points in between the datapoints are computed by cubic interpolation
		'''
		return interp1d(f_data, self._baseline_als(np.absolute(z_data),lam,p,niter=niter), kind='cubic')
		
	def baseline_func_phase(self,z_data,f_data,lam,p,niter=10):
		'''
		for this to work, you need to analyze a large part of the baseline
		tune lam and p until you get the desired result
		returns the baseline as a function
		the points in between the datapoints are computed by cubic interpolation
		'''
		return interp1d(f_data, self._baseline_als(np.angle(z_data),lam,p,niter=niter), kind='cubic')
		
	def fit_baseline_phase(self,z_data,lam,p,niter=10):
		'''
		for this to work, you need to analyze a large part of the baseline
		tune lam and p until you get the desired result
		'''
		return self._baseline_als(np.angle(z_data),lam,p,niter=niter)

	def GUIbaselinefit(self):
		'''
		A GUI to help you fit the baseline
		'''
		self.__lam = 1e6
		self.__p = 0.9
		niter = 10
		self.__baseline = self._baseline_als(np.absolute(self.z_data_raw),self.__lam,self.__p,niter=niter)
		import matplotlib.pyplot as plt
		from matplotlib.widgets import Slider
		fig, (ax0,ax1) = plt.subplots(nrows=2)
		plt.suptitle('Use the sliders to make the green curve match the baseline.')
		plt.subplots_adjust(left=0.25, bottom=0.25)
		l0, = ax0.plot(np.absolute(self.z_data_raw))
		l0b, = ax0.plot(np.absolute(self.__baseline))
		l1, = ax1.plot(np.absolute(self.z_data_raw/self.__baseline))
		ax0.set_ylabel('amp, rawdata vs. baseline')
		ax1.set_ylabel('amp, corrected')
		axcolor = 'lightgoldenrodyellow'
		axSmooth = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
		axAsym = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
		axbcorr = plt.axes([0.25, 0.05, 0.65, 0.03], axisbg=axcolor)
		sSmooth = Slider(axSmooth, 'Smoothness', 0.1, 10., valinit=np.log10(self.__lam),valfmt='1E%f')
		sAsym = Slider(axAsym, 'Asymmetry', 1e-4,0.99999, valinit=self.__p,valfmt='%f')
		sbcorr = Slider(axbcorr, 'vertical shift',0.7,1.1,valinit=1.)
		def update(val):
			self.__lam = 10**sSmooth.val
			self.__p = sAsym.val
			self.__baseline = sbcorr.val*self._baseline_als(np.absolute(self.z_data_raw),self.__lam,self.__p,niter=niter)
			l0.set_ydata(np.absolute(self.z_data_raw))
			l0b.set_ydata(np.absolute(self.__baseline))
			l1.set_ydata(np.absolute(self.z_data_raw/self.__baseline))
			fig.canvas.draw_idle()
		sSmooth.on_changed(update)
		sAsym.on_changed(update)
		sbcorr.on_changed(update)
		plt.show()
		self.z_data_raw /= self.__baseline
		plt.close()
		
		
		
		
		