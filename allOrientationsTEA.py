# finding correlations between TEA model parameters and galaxy properties

import numpy as np 
import matplotlib.pyplot as plt 
import fitsio
import os
import fsps
from timeit import default_timer as timer
import datetime
import scipy.optimize as opt
import dustModels as dm
from scipy.interpolate import CubicSpline
from scipy.interpolate import splrep, BSpline
from matplotlib import rc

def calcError(attFSPS):
        # this functions calculates the error between FSPS and catalog attenuation curves
        return np.sum(np.sqrt((attFSPS - currentMatchedAttCatalog)**2), axis=-1) / len(errorWave)

def TEAMin(x):
	# bump skew fixed to 6.95
	Av = x[0]
	dustIndex = x[1]
	bumpStrength = x[2]
	dustySpec, attenuationMags = dm.simpleTEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength)
	return calcError(attenuationMags)

def calcBumpStrengthTEA(Av, dustIndex, bumpStrength):
	bumpWaveMask = (np.log10(errorWave) > 3.25) & (np.log10(errorWave) < 3.6)
	dustySpec, attenuationMags = dm.simpleTEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength)
	bumpHeight = np.amax(attenuationMags[bumpWaveMask]) / Av
	linearFit = np.polyfit(np.log10(errorWave[~bumpWaveMask]), np.log10(attenuationMags[~bumpWaveMask]), 1)
	fit = np.poly1d(linearFit)
	x_fit = [np.amin(np.log10(errorWave)), np.amax(np.log10(errorWave))]
	y_fit = fit(x_fit)
	effPowerIndex = fit[1]
	return effPowerIndex, bumpHeight

def calcBumpStrengthKC(dustIndex):
	# imposing Av = 1, all light attenuated, no young attenuation
	bumpWaveMask = (np.log10(errorWave) > 3.3) & (np.log10(errorWave) < 3.4)
	bumpWaveMaskFit = (np.log10(errorWave) > 3.2) & (np.log10(errorWave) < 3.5)
	dustySpec, attenuationMags = dm.kriekAndConroy(errorWave, youngSpec, oldSpec, 1., dustIndex, 1., -0.7, 0., 0.)
	Av = attenuationMags[np.argmin((errorWave - 5500)**2)] 
	bumpHeight = np.amax(attenuationMags[bumpWaveMask]) / Av
	linearFit = np.polyfit(np.log10(errorWave[~bumpWaveMaskFit]), np.log10(attenuationMags[~bumpWaveMaskFit]), 1)
	fit = np.poly1d(linearFit)
	x_fit = [np.amin(np.log10(errorWave)), np.amax(np.log10(errorWave))]
	y_fit = fit(x_fit)
	effPowerIndex = fit[1]
	return effPowerIndex, bumpHeight

def TEA_KC_powerIndex_bumpStrength():
	bumpStrengthsKC = np.zeros(len(dustIndexKC))
	effPowerIndexKC = np.zeros(len(dustIndexKC)) # effectively shifted due to Calzetti component
	for i in range(len(dustIndexKC)):
		effPowerIndexKC[i], bumpStrengthsKC[i] = calcBumpStrengthKC(dustIndexKC[i])
	plt.figure(figsize=(10,8))
	labelBool = True
	plotBool = []
	for i in range(len(TEABestParams[:,0,0])):
		if np.amax(TEABestParams[i,:,0]) < 0.2:
			plotBool.append(False)
		else:
			plotBool.append(True)
	plotBool = np.array(plotBool, dtype=bool)
	bumpStrengthsTEA = np.zeros((len(TEABestParams[plotBool,0,0]), len(TEABestParams[0,:,0]))) # galaxy, orientation
	effPowerIndexTEA = np.zeros((len(TEABestParams[plotBool,0,0]), len(TEABestParams[0,:,0])))
	for i in range(len(TEABestParams[plotBool,0,0])):
		for j in range(len(bumpStrengthsTEA[0,:])):
			effPowerIndexTEA[i,j], bumpStrengthsTEA[i,j] = calcBumpStrengthTEA(TEABestParams[plotBool,j,0][i], TEABestParams[plotBool,j,1][i], TEABestParams[plotBool,j,2][i])
	plt.scatter(TEABestParams[plotBool,:,1], np.log10(bumpStrengthsTEA), alpha=0.5, s=50, color='green', label='TEA Input p', marker='o')
	plt.scatter(effPowerIndexTEA, np.log10(bumpStrengthsTEA), alpha=0.5, s=50, color='springgreen', label='TEA Effective p', marker='o')
	plt.scatter(dustIndexKC, np.log10(bumpStrengthsKC), alpha=0.5, s=50, color='blue', label='K&C Input p', marker='o')
	plt.scatter(effPowerIndexKC, np.log10(bumpStrengthsKC), alpha=0.5, s=50, color='cornflowerblue', label='K&C Effective p', marker='o')
	plt.xlabel('p', fontsize=28)
	plt.ylabel(r'$\log_{10}(Bump \; Height \, / \, A_{V})$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+'powerIndex_bumpHeight.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def matchWavelengths(wave, mags):
	newMags = np.zeros(len(errorWave))
	for j in range(len(errorWave)):
		waveMask = (np.log10(wave) > (np.log10(errorWave[j]) - logErrorSpacing/2)) & \
				   (np.log10(wave) <= (np.log10(errorWave[j]) + logErrorSpacing/2))
		newMags[j] = np.mean(mags[waveMask])
	return newMags

def plotKC():
	labelBool = True
	for i in range(len(TEABestParamsKC[:,0])):
		if labelBool:
			label1 = 'TEA'
			label2 = 'Kriek and Conroy'
		else:
			label1 = None
			label2 = None
		labelBool = False
		dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, TEABestParamsKC[i,0], TEABestParamsKC[i,1], TEABestParamsKC[i,2], TEABestParamsKC[i,3])
		dustySpecKC, KC = dm.kriekAndConroy(errorWave, youngSpec, oldSpec, 1., dustIndexKC[i], 1., -0.7, 0., 0.)
		plt.plot(np.log10(errorWave), attenuationMags, label=label1, alpha=0.5, linewidth=1, linestyle='dashed', color='green')
		plt.plot(np.log10(errorWave), KC, label=label2, alpha=0.5, linewidth=1, linestyle='solid', color='blue')
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=28)
	plt.ylabel(r'$A_{\lambda}$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+'/TEA_KC_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def powerIndexBumpStrength():
	x = []
	y = []
	plt.figure(figsize=(10,8))
	labelBool = True
	for i in range(len(singleNames[minAvMask])):
		if np.amax(TEABestParams[i,:,0]) < 0.2:
			continue
		if labelBool:
			label = 'NIHAO-SKIRT-Catalog'
		else:
			label = None
		labelBool = False
		plt.scatter(TEABestParams[i,:,1], np.log10(TEABestParams[i,:,2] / TEABestParams[i,:,0]), alpha=0.5, s=60, color='green', label=label)
		x.append(TEABestParams[i,:,1])
		y.append(np.log10(TEABestParams[i,:,2] / TEABestParams[i,:,0]))
	linearFit = np.polyfit(np.ravel(x), np.ravel(y), 1)
	fit = np.poly1d(linearFit)
	print('fit values:', fit)
	x_fit = [np.amin(x), np.amax(x)]
	y_fit = fit(x_fit)
	plt.plot(x_fit, y_fit, color='green', alpha=0.5, label="\N{MINUS SIGN}"+"{:.3f}".format(abs(fit[0]))+" \N{MINUS SIGN} "+"{:.3f}".format(abs(fit[1]))+'p', linewidth=4)
	plt.xlabel('p', fontsize=28)
	plt.ylabel(r'$\log_{10}(b_{UV} \, / \, A_{V})$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+'powerIndex_bumpStrength.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def AvPowerIndex():
	x = []
	y = []
	plt.figure(figsize=(10,8))
	labelBool = True
	for i in range(len(singleNames[minAvMask])):
		if np.amax(TEABestParams[i,:,0]) < 0.05:
			continue
		if labelBool:
			label = 'NIHAO-SKIRT-Catalog'
		else:
			label = None
		labelBool = False
		plt.scatter(np.log10(TEABestParams[i,:,0]), TEABestParams[i,:,1], alpha=0.5, s=60, color='green', label=label)
		x.append(np.log10(TEABestParams[i,:,0]))
		y.append(TEABestParams[i,:,1])
	linearFit = np.polyfit(np.ravel(x), np.ravel(y), 1)
	fit = np.poly1d(linearFit)
	print('fit values:', fit)
	x_fit = [np.amin(x), np.amax(x)]
	y_fit = fit(x_fit)
	plt.plot(x_fit, y_fit, color='green', alpha=0.5, label="\N{MINUS SIGN}"+"{:.3f}".format(abs(fit[0]))+" + "+"{:.3f}".format(abs(fit[1]))+r'$\, \log_{10}(A_{V})$', linewidth=4)
	plt.xlabel(r'$\log_{10}(A_{V})$', fontsize=28)
	plt.ylabel('p',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+'Av_powerIndex.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def AvBumpStrength():
	plt.figure(figsize=(10,8))
	for i in range(len(singleNames[minAvMask])):
		plt.scatter(TEABestParams[i,:,0], TEABestParams[i,:,2], alpha=0.5, s=50)
	plt.xlabel('Av', fontsize=28)
	plt.ylabel('Bump Strength',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.savefig(plotPath+'Av_bumpStrength.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def massAv():
	plt.figure(figsize=(10,8))
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		plt.scatter(np.log10(trueMasses), np.log10(TEABestParams[i,:,0]), alpha=0.5, s=50)
	plt.xlabel('Log(True Stellar Mass)', fontsize=28)
	plt.ylabel('Log(Av)',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.savefig(plotPath+'trueMass_Av.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def massBumpStrength():
	plt.figure(figsize=(10,8))
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		plt.scatter(np.log10(trueMasses), np.log10(TEABestParams[i,:,2]), alpha=0.5, s=50)
	plt.xlabel('Log(True Stellar Mass)', fontsize=28)
	plt.ylabel('Log(Bump Strength)',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.savefig(plotPath+'trueMass_bumpStrength.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def axisRatioPowerIndex():
	plt.figure(figsize=(10,8))
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		plt.scatter(trueAxisRatios, TEABestParams[i,:,1], alpha=0.5, s=50)
	plt.xlabel('Axis Ratio', fontsize=28)
	plt.ylabel('Power Index',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.savefig(plotPath+'axisRatio_powerIndex.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def axisRatioBumpStrength():
	plt.figure(figsize=(10,8))
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		plt.scatter(trueAxisRatios, np.log10(TEABestParams[i,:,2]), alpha=0.5, s=50)
	plt.xlabel('Axis Ratio', fontsize=28)
	plt.ylabel('Log(Bump Strength)',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.savefig(plotPath+'axisRatio_bumpStrength.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def SFRPowerIndex():
	plt.figure(figsize=(10,8))
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		SFRs = SFR[nameMaskFull]
		plt.scatter(np.log10(SFRs), TEABestParams[i,:,1], alpha=0.5, s=50)
	plt.xlabel('Log(SFR)', fontsize=28)
	plt.ylabel('Power Index',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.savefig(plotPath+'SFR_powerIndex.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def SFRBumpStrength():
	plt.figure(figsize=(10,8))
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		SFRs = SFR[nameMaskFull]
		plt.scatter(np.log10(SFRs), np.log10(TEABestParams[i,:,2]), alpha=0.5, s=50)
	plt.xlabel('Log(SFR)', fontsize=28)
	plt.ylabel('Log(Power Index)',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.savefig(plotPath+'SFR_bumpStrength.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def axisRatioPowerIndexSeparate():
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		if np.amin(trueAxisRatios) >= 0.3:
			continue
		plt.figure(figsize=(10,8))
		plt.scatter(trueAxisRatios, TEABestParams[i,:,1], alpha=0.75, s=100, color='k')
		plt.xlabel('Axis Ratio', fontsize=28)
		plt.ylabel('Power Index',fontsize=28)
		plt.xticks(fontsize=28)
		plt.yticks(fontsize=28)
		os.system('mkdir -p '+plotPath+'axisRatio_powerIndex/')
		plt.savefig(plotPath+'axisRatio_powerIndex/'+galaxy+'_axisRatio_powerIndex.png', 
					dpi=300, bbox_inches='tight', pad_inches=0.5)
		plt.close()

def axisRatioBumpStrengthSeparate():
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		if np.amin(trueAxisRatios) >= 0.3:
			continue
		plt.figure(figsize=(10,8))
		plt.scatter(trueAxisRatios, np.log10(TEABestParams[i,:,2] / TEABestParams[i,:,0]), alpha=0.75, s=100, color='k')
		plt.xlabel('Axis Ratio', fontsize=28)
		plt.ylabel('Log(Bump Strength / Av)',fontsize=28)
		plt.xticks(fontsize=28)
		plt.yticks(fontsize=28)
		os.system('mkdir -p '+plotPath+'axisRatio_bumpStrength/')
		plt.savefig(plotPath+'axisRatio_bumpStrength/'+galaxy+'_axisRatio_bumpStrength.png', 
					dpi=300, bbox_inches='tight', pad_inches=0.5)
		plt.close()

def AvBumpStrengthSeparate():
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		if np.amin(trueAxisRatios) >= 0.3:
			continue
		plt.figure(figsize=(10,8))
		plt.scatter(np.log10(TEABestParams[i,:,0]), np.log10(TEABestParams[i,:,2] / TEABestParams[i,:,0]), alpha=0.75, s=100, color='k')
		plt.xlabel('Log(Av)', fontsize=28)
		plt.ylabel('Log(Bump Strength / Av)',fontsize=28)
		plt.xticks(fontsize=28)
		plt.yticks(fontsize=28)
		os.system('mkdir -p '+plotPath+'Av_bumpStrength/')
		plt.savefig(plotPath+'Av_bumpStrength/'+galaxy+'_Av_bumpStrength.png', 
					dpi=300, bbox_inches='tight', pad_inches=0.5)
		plt.close()

def powerIndexBumpStrengthSeparate():
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		nameMaskFull = names == galaxy
		trueMasses = stellarMass[nameMaskFull]
		trueAxisRatios = axisRatios[nameMaskFull]
		if np.amin(trueAxisRatios) >= 0.3:
			continue
		plt.figure(figsize=(10,8))
		plt.scatter(TEABestParams[i,:,1], np.log10(TEABestParams[i,:,2]), alpha=0.75, s=100, color='k')
		plt.xlabel('Power Index', fontsize=28)
		plt.ylabel('Log(Bump Strength)',fontsize=28)
		plt.xticks(fontsize=28)
		plt.yticks(fontsize=28)
		os.system('mkdir -p '+plotPath+'powerIndex_bumpStrength/')
		plt.savefig(plotPath+'powerIndex_bumpStrength/'+galaxy+'_powerIndex_bumpStrength.png', 
					dpi=300, bbox_inches='tight', pad_inches=0.5)
		plt.close()

def plotTEA_varyParams():
	plotWave = np.logspace(np.log10(1e3), np.log10(1e4), num=200, endpoint=True)
	youngSpec = np.zeros(len(plotWave), dtype=np.float32)
	oldSpec = np.ones(len(plotWave), dtype=np.float32)
	plt.figure(figsize=(10,8))
	dustySpec, attenuationMags = dm.TEA(plotWave, youngSpec, oldSpec, 0.5, -0.8, 0.)
	plt.plot(np.log10(plotWave), attenuationMags, alpha=0.7, linewidth=3, linestyle=(0,(5,0.5)), color='k', label=r'$A_{V}=0.5, \, p= \hspace{-0.25} -0.8, \, b_{UV}=0$')
	dustySpec, attenuationMags = dm.TEA(plotWave, youngSpec, oldSpec, 0.5, -1., 0.)
	plt.plot(np.log10(plotWave), attenuationMags, alpha=0.7, linewidth=3, linestyle=(0,(1,5)), color='k', label=r'$A_{V}=0.5, \, p= \hspace{-0.25} -1, \, b_{UV}=0$')
	dustySpec, attenuationMags = dm.TEA(plotWave, youngSpec, oldSpec, 1., -0.8, 0.)
	plt.plot(np.log10(plotWave), attenuationMags, alpha=0.7, linewidth=3, linestyle='solid', color='k', label=r'$A_{V}=1, \, p= \hspace{-0.25} -0.8, \, b_{UV}=0$')
	dustySpec, attenuationMags = dm.TEA(plotWave, youngSpec, oldSpec, 1, -0.8, 0.5)
	plt.plot(np.log10(plotWave), attenuationMags, alpha=0.7, linewidth=3, linestyle='dashed', color='k', label=r'$A_{V}=1, \, p= \hspace{-0.25} -0.8, \, b_{UV}=0.5$')
	dustySpec, attenuationMags = dm.TEA(plotWave, youngSpec, oldSpec, 1, -0.8, 1.)
	plt.plot(np.log10(plotWave), attenuationMags, alpha=0.7, linewidth=3, linestyle=(0,(1,1)), color='k', label=r'$A_{V}=1, \, p= \hspace{-0.25} -0.8, \, b_{UV}=1$')
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=28)
	plt.ylabel(r'$A_{\lambda}$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=16)
	plt.savefig(plotPath+'plotTEA_varyParams.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

fit = False
fitKC = False # fit TEA model to kriek and conroy curves
makePlots = True

plotPath = 'allOrientationsTEA/Plots/'
os.system('mkdir -p '+plotPath)

# Load fits file
galaxies = fitsio.read('~/NIHAO-SKIRT-Catalog/nihao-integrated-seds.fits', ext='GALAXIES') # one per galaxy
summary = fitsio.read('~/NIHAO-SKIRT-Catalog/nihao-integrated-seds.fits', ext='SUMMARY') # 10 per galaxy (one per viewing orientation)
wave = fitsio.read('~/NIHAO-SKIRT-Catalog/nihao-integrated-seds.fits', ext='WAVE')
spectrum = fitsio.read('~/NIHAO-SKIRT-Catalog/nihao-integrated-seds.fits', ext='SPEC')
spectrum_nodust = fitsio.read('~/NIHAO-SKIRT-Catalog/nihao-integrated-seds.fits', ext='SPECNODUST')
attenuation_wave = fitsio.read('~/NIHAO-SKIRT-Catalog/nihao-integrated-seds.fits', ext='ATTENUATION_WAVE')
attenuation_mags = fitsio.read('~/NIHAO-SKIRT-Catalog/nihao-integrated-seds.fits', ext='ATTENUATION_MAGS')

singleNames = galaxies['name'] # one per galaxy
singleStellarMass = galaxies['stellar_mass'] # one per galaxy, in M_sun
names = summary['name']
stellarMass = summary['stellar_mass'] # in solar mass
SFR = summary['sfr'] # in M_sun per year
dustMass = summary['dust_mass'] # in M_sun
axisRatios = summary['axis_ratio']
Av = summary['Av'] # in magnitudes
bands = summary['bands'][0]
flux = summary['flux'] # in Jansky
flux_noDust = summary['flux_nodust'] # in Jansky
attEnergy = summary['attenuated_energy'] # in 10^-23 erg s^-1 cm^-2
emitEnergy = summary['emitted_energy'] # in 10^-23 erg s^-1 cm^-2
SFH = galaxies['sfh'] # in M_sun (as a function of ages)
CEH = galaxies['ceh'] # in M_sun  (as a function of ages and metals)
ages = galaxies['ages'] # age bins in yeras for SFH and CEH
metals = galaxies['metals'] # metallicity bins for CEH

# mask to wavelengths that contribute to error calculation
waveMaskCatalog = (attenuation_wave >= 1e3) & (attenuation_wave <= 1e4)
attenuation_wave = attenuation_wave[waveMaskCatalog]
attenuation_mags = attenuation_mags[:, waveMaskCatalog]

# calculate error at errorWave wavelengths
errorWave = np.logspace(np.log10(1e3), np.log10(1e4), num=75, endpoint=True)
logErrorSpacing = np.log10(errorWave[1]) - np.log10(errorWave[0])
errorWaveExt = errorWave.copy()
for i in range(1):
	errorWaveExt = np.append(errorWaveExt, 10**(np.log10(errorWaveExt[-1])+logErrorSpacing)) # for Cardelli

logErrorSpacing = np.log10(errorWave[1]) - np.log10(errorWave[0])

matchedAttCatalog = np.zeros((len(attenuation_mags[:,0]), len(errorWave)))
for i in range(len(attenuation_mags[:,0])):
	matchedAttCatalog[i,:] = matchWavelengths(attenuation_wave, attenuation_mags[i, :])

minAvMask = []
for i in range(len(singleNames)):
	nameMask = names == singleNames[i]
	#if (np.amin(Av[nameMask]) > 0.2):
	if (np.amax(Av[nameMask]) > 0.): 
		minAvMask.append(True)
	else:
		minAvMask.append(False)

# TEA model is single component, don't need spectra to calculate attenuation curves
youngSpec = np.zeros(len(errorWave), dtype=np.float32)
oldSpec = np.ones(len(errorWave), dtype=np.float32)

if fit:
	maxIter = 2000
	TEABestParams = np.zeros((len(singleNames[minAvMask]), 10, 3))
	for i in range(len(singleNames[minAvMask])): 
		galaxy = singleNames[minAvMask][i]
		print('fitting', galaxy)
		nameMaskFull = names == galaxy
		for j in range(10):
			trueAv = Av[nameMaskFull][j]
			trueAxisRatio = axisRatios[nameMaskFull][j]
			# find best parameters
			currentMatchedAttCatalog = matchedAttCatalog[nameMaskFull][j]
			TEAResult = opt.dual_annealing(TEAMin, maxiter=maxIter,
						bounds=((0.,10.), (-5., 0.), (0.0,10)))
			TEABestParams[i,j,:] = TEAResult.x
	os.system('mkdir -p allOrientationsTEA/bestParams/')
	np.save('allOrientationsTEA/bestParams/TEABestParams.npy', TEABestParams)

dustIndexKC = np.array([-1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.])

if makePlots:
	print('making plots')
	TEABestParams = np.load('allOrientationsTEA/bestParams/TEABestParams.npy')
	TEA_KC_powerIndex_bumpStrength() 
	powerIndexBumpStrength()
	AvPowerIndex()
	#AvBumpStrength()
	#AvBumpSkew()
	#powerIndexBumpSkew()
	#bumpStregnthBumpSkew()
	#massAv()
	#massBumpStrength()
	#axisRatioPowerIndex()
	#axisRatioBumpStrength()
	#SFRPowerIndex()
	#SFRBumpStrength()
	#axisRatioPowerIndexSeparate()
	#axisRatioBumpStrengthSeparate()
	#AvBumpStrengthSeparate()
	#powerIndexBumpStrengthSeparate()
	plotTEA_varyParams()
