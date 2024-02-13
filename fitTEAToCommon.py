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

def youngAndOldSEDs(galaxy, age):
	os.system('mkdir -p youngAndOldSEDs/'+galaxy+'/'+age+'/') 
	nameMask = singleNames == galaxy
	nameMaskFull = names == galaxy
	faceIndex = np.argmax(axisRatios[nameMaskFull])
	xbins = ages[nameMask, :][0]
	ybins = metals[nameMask, :][0]
	# change bins from left edge to center
	xRightEdge = 10**(np.log10(xbins[1]) - np.log10(xbins[0]) + np.log10(xbins[-1]))
	xbins = np.append(xbins, xRightEdge)
	yRightEdge = 10**(np.log10(ybins[1]) - np.log10(ybins[0]) + np.log10(ybins[-1]))
	ybins = np.append(ybins, yRightEdge)
	xbins = (xbins[0:-1] + xbins[1:]) / 2
	ybins = (ybins[0:-1] + ybins[1:]) / 2
	ageBins, metalBins = np.meshgrid(xbins, ybins)
	ageBins /= 1e9 # convert to Gyrs
	masses = CEH[nameMask, :, :][0]
	flatAges = ageBins.flatten()
	flatMetals = metalBins.flatten()
	flatMasses = masses.flatten(order='F')
	flatMetals = np.log10(flatMetals / 0.019) # change from Z to log10(Z / Zsun)
	# mask out bins with no mass
	massMask = flatMasses > 0
	flatAges = flatAges[massMask]
	flatMetals = flatMetals[massMask]
	flatMasses = flatMasses[massMask]
	sp = fsps.StellarPopulation(sfh=0, compute_vega_mags=False, zcontinuous=1, 
								dust_type=0, dust1=0, dust2=0, imf_type=1, add_agb_dust_model=False)
	waveFSPS, specFSPS = sp.get_spectrum(tage=0.1, peraa=False) # tage doesn't matter here
	waveMask = (waveFSPS >= 1e3) & (waveFSPS <= 2e5)
	waveMaskCatalog = (wave >= 1e3) & (wave <= 2e5)
	youngSpec = np.zeros(len(specFSPS))
	oldSpec = np.zeros(len(specFSPS))
	for i in range(len(flatAges)):
		sp.params['logzsol'] = flatMetals[i]
		waveFSPS, specFSPS = sp.get_spectrum(tage = flatAges[i], peraa=False)
		if flatAges[i] <= float(age):
			youngSpec += specFSPS * flatMasses[i]
		else:
			oldSpec += specFSPS * flatMasses[i]
	d = 3.086e24 # 100 Mpc in meters
	youngSpec *= 3.83e26 # Lsun/Hz to Watts/Hz
	youngSpec = (youngSpec / (4 * np.pi * d**2)) * 1e26 # convert to Jansky
	oldSpec *= 3.83e26 # Lsun/Hz to Watts/Hz
	oldSpec = (oldSpec / (4 * np.pi * d**2)) * 1e26 # convert to Jansky
	np.save('youngAndOldSEDs/waveFSPS.npy', waveFSPS)
	np.save('youngAndOldSEDs/'+galaxy+'/'+age+'/young.npy', youngSpec)
	np.save('youngAndOldSEDs/'+galaxy+'/'+age+'/old.npy', oldSpec)

def getAgeThresholds(galaxy, ageMin, ageMax):
	# find bin edges for galaxy within ageMin and ageMax in Gyrs
	nameMask = singleNames == galaxy
	xbins = ages[nameMask, :][0]
	# change bins from left edge to center
	xRightEdge = 10**(np.log10(xbins[1]) - np.log10(xbins[0]) + np.log10(xbins[-1]))
	xbins = np.append(xbins, xRightEdge)
	xbins = (xbins[0:-1] + xbins[1:]) / 2
	xbins /= 1e9 # conver to Gyrs
	ageMask = (xbins >= ageMin) & (xbins <= ageMax)
	return xbins[ageMask]

def getClosestAgeThreshold(galaxy, age):
	# find bin edges for galaxy within ageMin and ageMax in Gyrs
	nameMask = singleNames == galaxy
	xbins = ages[nameMask, :][0]
	xbins /= 1e9 # conver to Gyrs
	index = np.argmin((xbins - age)**2)
	return xbins[index]

def chiSquare(attWaveFSPS, attWaveCatalog, attFSPS, attCatalog):
	matchedAttFSPS = np.zeros(len(attWaveCatalog))
	for i in range(len(attWaveCatalog)):
		matchedInd = np.argmin((attWaveCatalog[i] - attWaveFSPS)**2)
		matchedAttFSPS[i] = attFSPS[matchedInd]
	return np.sum((matchedAttFSPS - attCatalog)**2 / attCatalog) / len(attWaveCatalog)

def calcError(attFSPS):
        # this functions calculates the error between FSPS and catalog attenuation curves
        return np.sum(np.sqrt((attFSPS - currentMatchedAttCatalog)**2), axis=-1) / len(errorWave)

def calcTEAToCommonError(attFSPS):
        # this functions calculates the error between TEA and commonly used dust models
        return np.sum(np.sqrt((attFSPS - attCommon)**2), axis=-1) / len(errorWave)

# old version with 4 free parameters
#def TEAToCommonMin(x):
#	Av = x[0]
#	dustIndex = x[1]
#	bumpStrength = x[2]
#	bumpSkew = x[3]
#	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength, bumpSkew)
#	return calcTEAToCommonError(attenuationMags)

def TEAToCommonMin(x):
	Av = x[0]
	dustIndex = x[1]
	bumpStrength = x[2]
	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength)
	return calcTEAToCommonError(attenuationMags)

def calzettiMin(x):
	Av = x[0]
	fracNoDust = x[1]
	dustySpec, attenuationMags = dm.calzetti(errorWave, youngSpec+oldSpec, Av, fracNoDust)
	return calcError(attenuationMags)

def calzettiMinNoFrac(x):
	fracNoDust = 0
	Av = x
	dustySpec, attenuationMags = dm.calzetti(errorWave, youngSpec+oldSpec, Av, fracNoDust)
	return calcError(attenuationMags)

def plotCalzetti(x, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = 'CalzettiNoFrac'
	else:
		name = 'Calzetti'
	Av = x[0]
	fracNoDust = x[1]
	dustySpec, attenuationMags = dm.calzetti(errorWave, youngSpec+oldSpec, Av, fracNoDust)
	os.system('mkdir -p '+plotPath+'best'+name+'/')
	plt.figure(figsize=(10,8))
	plt.plot(np.log10(errorWave), attenuationMags, label='Calzetti', alpha=0.5, linewidth=3)
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=3)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=28)
	plt.ylabel(r'$A_{\lambda}$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=28)
	plt.savefig(plotPath+'best'+name+'/'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def bestCalzetti(x, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = 'CalzettiNoFrac'
	else:
		name = 'Calzetti'
	Av = x[0]
	fracNoDust = x[1]
	dustySpec, attenuationMags = dm.calzetti(errorWave, youngSpec+oldSpec, Av, fracNoDust)
	return attenuationMags

def cardelliMin(x):
	Av = x[0]
	mwr = x[1]
	uvb = x[2]
	AvYoung = x[3]
	dustIndexYoung = x[4]
	fracNoDust = x[5]
	fracNoDustYoung = x[6]
	dustySpec, attenuationMags = dm.cardelli(errorWaveExt, youngSpecExt, oldSpecExt, Av, mwr, uvb, AvYoung, 
			 					 dustIndexYoung, fracNoDust, fracNoDustYoung)
	dustySpec = dustySpec[:-1]
	attenuationMags = attenuationMags[:-1]
	return calcError(attenuationMags)

def cardelliMinNoFrac(x):
	fracNoDust = 0
	fracNoDustYoung = 0
	Av = x[0]
	mwr = x[1]
	uvb = x[2]
	AvYoung = x[3]
	dustIndexYoung = x[4]
	dustySpec, attenuationMags = dm.cardelli(errorWaveExt, youngSpecExt, oldSpecExt, Av, mwr, uvb, AvYoung, 
			 					 dustIndexYoung, fracNoDust, fracNoDustYoung)
	dustySpec = dustySpec[:-1]
	attenuationMags = attenuationMags[:-1]
	return calcError(attenuationMags)

def plotCardelli(x, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = 'CardelliNoFrac'
	else:
		name = 'Cardelli'
	Av = x[0]
	mwr = x[1]
	uvb = x[2]
	AvYoung = x[3]
	dustIndexYoung = x[4]
	fracNoDust = x[5]
	fracNoDustYoung = x[6]
	dustySpec, attenuationMags = dm.cardelli(errorWaveExt, youngSpecExt, oldSpecExt, Av, mwr, uvb, AvYoung, 
			 					 dustIndexYoung, fracNoDust, fracNoDustYoung)
	dustySpec = dustySpec[:-1]
	attenuationMags = attenuationMags[:-1]
	os.system('mkdir -p '+plotPath+'best'+name+'/')
	plt.figure(figsize=(10,8))
	plt.plot(np.log10(errorWave), attenuationMags, label='Cardelli', alpha=0.5, linewidth=3)
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=3)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=28)
	plt.ylabel(r'$A_{\lambda}$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=28)
	plt.savefig(plotPath+'best'+name+'/'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def bestCardelli(x, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = 'CardelliNoFrac'
	else:
		name = 'Cardelli'
	Av = x[0]
	mwr = x[1]
	uvb = x[2]
	AvYoung = x[3]
	dustIndexYoung = x[4]
	fracNoDust = x[5]
	fracNoDustYoung = x[6]
	dustySpec, attenuationMags = dm.cardelli(errorWaveExt, youngSpecExt, oldSpecExt, Av, mwr, uvb, AvYoung, 
			 					 dustIndexYoung, fracNoDust, fracNoDustYoung)
	dustySpec = dustySpec[:-1]
	attenuationMags = attenuationMags[:-1]
	return attenuationMags

def powerLawMin(x):
	Av = x[0]
	dustIndex = x[1]
	AvYoung = x[2]
	dustIndexYoung = x[3]
	fracNoDust = x[4]
	fracNoDustYoung = x[5]
	dustySpec, attenuationMags = dm.powerLaw(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	return calcError(attenuationMags)

def powerLawMinNoFrac(x):
	fracNoDust = 0
	fracNoDustYoung = 0
	Av = x[0]
	dustIndex = x[1]
	AvYoung = x[2]
	dustIndexYoung = x[3]
	dustySpec, attenuationMags = dm.powerLaw(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	return calcError(attenuationMags)

def plotPowerLaw(x, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = 'PowerLawNoFrac'
	else:
		name = 'PowerLaw'
	Av = x[0]
	dustIndex = x[1]
	AvYoung = x[2]
	dustIndexYoung = x[3]
	fracNoDust = x[4]
	fracNoDustYoung = x[5]
	dustySpec, attenuationMags = dm.powerLaw(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	#os.system('mkdir -p '+plotPath+'bestPowerLaw/'+galaxy+'/')
	os.system('mkdir -p '+plotPath+'best'+name+'/')
	plt.figure(figsize=(10,8))
	plt.plot(np.log10(errorWave), attenuationMags, label='Power Law', alpha=0.5, linewidth=3)
	#plt.plot(np.log10(attenuation_wave), attenuation_mags[nameMaskFull][faceIndex], label='Catalog', alpha=0.5)
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=3)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=28)
	plt.ylabel(r'$A_{\lambda}$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=28)
	plt.savefig(plotPath+'best'+name+'/'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def bestPowerLaw(x, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = 'PowerLawNoFrac'
	else:
		name = 'PowerLaw'
	Av = x[0]
	dustIndex = x[1]
	AvYoung = x[2]
	dustIndexYoung = x[3]
	fracNoDust = x[4]
	fracNoDustYoung = x[5]
	dustySpec, attenuationMags = dm.powerLaw(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	return attenuationMags

def kriekAndConroyMin(x):
	Av = x[0]
	dustIndex = x[1]
	AvYoung = x[2]
	dustIndexYoung = x[3]
	fracNoDust = x[4]
	fracNoDustYoung = x[5]
	dustySpec, attenuationMags = dm.kriekAndConroy(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	return calcError(attenuationMags)

def kriekAndConroyMinNoFrac(x):
	fracNoDust = 0
	fracNoDustYoung = 0
	Av = x[0]
	dustIndex = x[1]
	AvYoung = x[2]
	dustIndexYoung = x[3]
	dustySpec, attenuationMags = dm.kriekAndConroy(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	return calcError(attenuationMags)

def plotKriekAndConroy(x, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = 'KriekAndConroyNoFrac'
	else:
		name = 'KriekAndConroy'
	Av = x[0]
	dustIndex = x[1]
	AvYoung = x[2]
	dustIndexYoung = x[3]
	fracNoDust = x[4]
	fracNoDustYoung = x[5]
	dustySpec, attenuationMags = dm.kriekAndConroy(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	#os.system('mkdir -p '+plotPath+'bestKriekAndConroy/'+galaxy+'/')
	os.system('mkdir -p '+plotPath+'best'+name+'/')
	plt.figure(figsize=(10,8))
	plt.plot(np.log10(errorWave), attenuationMags, label='Kriek and Conroy', alpha=0.5, linewidth=3)
	#plt.plot(np.log10(attenuation_wave), attenuation_mags[nameMaskFull][faceIndex], label='Catalog', alpha=0.5)
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=3)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=28)
	plt.ylabel(r'$A_{\lambda}$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=28)
	plt.savefig(plotPath+'best'+name+'/'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def bestKriekAndConroy(x, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = 'KriekAndConroyNoFrac'
	else:
		name = 'KriekAndConroy'
	Av = x[0]
	dustIndex = x[1]
	AvYoung = x[2]
	dustIndexYoung = x[3]
	fracNoDust = x[4]
	fracNoDustYoung = x[5]
	dustySpec, attenuationMags = dm.kriekAndConroy(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	return attenuationMags

def TEAMin(x):
	Av = x[0]
	dustIndex = x[1]
	bumpStrength = x[2]
	bumpSkew = x[3]
	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength, bumpSkew)
	return calcError(attenuationMags)

def plotTEA(x, orientation):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	Av = x[0]
	dustIndex = x[1]
	bumpStrength = x[2]
	bumpSkew = x[3]
	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength, bumpSkew)
	os.system('mkdir -p '+plotPath+'bestTEA/')
	plt.figure(figsize=(10,8))
	plt.plot(np.log10(errorWave), attenuationMags, label='TEA', alpha=0.5, linewidth=3)
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=3)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=28)
	plt.ylabel(r'$A_{\lambda}$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=28)
	plt.savefig(plotPath+'bestTEA/TEA_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

# old version with 4 free parameters
#def bestTEA(x):
#	Av = x[0]
#	dustIndex = x[1]
#	bumpStrength = x[2]
#	bumpSkew = x[3]
#	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength, bumpSkew)
#	return attenuationMags

def bestTEA(x):
	Av = x[0]
	dustIndex = x[1]
	bumpStrength = x[2]
	dustySpec, attenuationMags = dm.simpleTEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength)
	return attenuationMags

def plotErrors(fullErrors, frac):
	if frac == 'noFrac':
		folder = 'TEAToCommon/noFrac_'
	else:
		folder = 'TEAToCommon/'
	os.system('mkdir -p '+plotPath+folder)
	plt.figure(figsize=(10,8))
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,0,0]), label='Calzetti Face-on', color='pink', marker='o', alpha=0.7, s=150, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,0,1]), label='Calzetti Edge-on', color='pink', marker='s', alpha=0.7, s=150, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,1,0]), label='Cardelli Face-on', color='sandybrown', marker='o', alpha=0.7, s=150, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,1,1]), label='Cardelli Edge-on', color='sandybrown', marker='s', alpha=0.7, s=150, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,2,0]), label='Power Law Face-on', color='tomato', marker='o', alpha=0.7, s=150, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,2,1]), label='Power Law Edge-on', color='tomato', marker='s', alpha=0.7, s=150, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,3,0]), label='Kriek and Conroy Face-on', color='cornflowerblue', marker='o', alpha=0.7, s=150, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,3,1]), label='Kriek and Conroy Edge-on', color='cornflowerblue', marker='s', alpha=0.7, s=150, linewidth=0)
	plt.xlabel(r'$\log_{10}(Stellar \; Mass \, / \, M_{\odot})$', fontsize=28)
	plt.ylabel(r'$\log_{10}(RMSE)$',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=10)
	plt.savefig(plotPath+folder+'TEAToCommonErrors.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def plotErrorsSeparate(fullErrors, frac):
	if frac == 'noFrac':
		folder = 'TEAToCommon/noFrac_'
	else:
		folder = 'TEAToCommon/'
	size = 200
	plt.figure(figsize=(10,8))
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,0,0]), label='Calzetti', color='pink', marker='s', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,1,0]), label='Cardelli', color='sandybrown', marker='^', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,2,0]), label='Power Law', color='tomato', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,3,0]), label='Kriek and Conroy', color='cornflowerblue', marker='D', alpha=0.7, s=size, linewidth=0)
	plt.xlabel(r'$\log_{10}(Stellar \; Mass \, / \, M_{\odot})$', fontsize=28)
	plt.ylabel(r'$\log_{10}(RMSE)$',fontsize=28)
	plt.title('Face-on',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+folder+'TEAToCommon_face-on_errors.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()
	plt.figure(figsize=(10,8))
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,0,1]), label='Calzetti', color='pink', marker='s', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,1,1]), label='Cardelli', color='sandybrown', marker='^', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,2,1]), label='Power Law', color='tomato', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,3,1]), label='Kriek and Conroy', color='cornflowerblue', marker='D', alpha=0.7, s=size, linewidth=0)
	plt.xlabel(r'$\log_{10}(Stellar \; Mass \, / \, M_{\odot})$', fontsize=28)
	plt.ylabel(r'$\log_{10}(RMSE)$',fontsize=28)
	plt.title('Edge-on',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+folder+'TEAToCommon_edge-on_errors.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def matchWavelengths(wave, mags):
	newMags = np.zeros(len(errorWave))
	for j in range(len(errorWave)):
		waveMask = (np.log10(wave) > (np.log10(errorWave[j]) - logErrorSpacing/2)) & \
				   (np.log10(wave) <= (np.log10(errorWave[j]) + logErrorSpacing/2))
		newMags[j] = np.mean(mags[waveMask])
	return newMags

def matchWavelengthsExt(wave, mags):
	# extended wavelength grid for Cardelli
	newMags = np.zeros(len(errorWaveExt))
	for j in range(len(errorWaveExt)):
		waveMask = (np.log10(wave) > (np.log10(errorWaveExt[j]) - logErrorSpacing/2)) & \
				   (np.log10(wave) <= (np.log10(errorWaveExt[j]) + logErrorSpacing/2))
		newMags[j] = np.mean(mags[waveMask])
	return newMags

def plotCurves(model, orientation, frac):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	if frac == 'noFrac':
		name = model+'NoFrac'
	else:
		name = model
	if model == 'KriekAndConroy':
		label = 'Kriek and Conroy'
	elif model == 'PowerLaw':
		label = 'Power Law'
	else:
		label = model
	os.system('mkdir -p '+plotPath+'TEAToCommon/TEAToCommon_'+name+'/')
	plt.figure(figsize=(10,8))
	plt.plot(np.log10(errorWave), attTEA, label='TEA', alpha=0.5, linewidth=4.5, linestyle='dashed')
	plt.plot(np.log10(errorWave), attCommon, label=label, alpha=0.5, linewidth=4.5)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=42)
	plt.ylabel(r'$A_{\lambda}$',fontsize=42)
	plt.xticks(fontsize=42)
	plt.yticks(fontsize=42)
	plt.legend(fontsize=30)
	plt.savefig(plotPath+'TEAToCommon/TEAToCommon_'+name+'/TEA'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #

storeSSPs = False
fit = False
makePlots = True

youngAndOldSEDsPath = 'youngAndOldSEDs/'

plotPath = 'Plots/'
os.system('mkdir -p '+plotPath)

#ageThresholdRange = [0.009, 0.01] # in Gyrs
ageThreshold = 0.01

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

# store young and old FSPS spectra for single age threshold
if storeSSPs:
	for i in range(len(singleNames[minAvMask])): # only loop over galaxies with minimum Av > 0.2
		#ageThresholds = getAgeThresholds(singleNames[minAvMask][i], ageThresholdRange[0], ageThresholdRange[1])
		closestAge = getClosestAgeThreshold(singleNames[minAvMask][i], ageThreshold)
		#print('Number of ages for '+singleNames[minAvMask][i]+': '+str(len(ageThresholds)))
		start = timer()
		youngAndOldSEDs(singleNames[minAvMask][i], str(closestAge))
		#plotDustFreeSEDs(singleNames[i], str(ageThresholds[j]))
		print('Time for '+singleNames[minAvMask][i]+': '+str(datetime.timedelta(seconds=(timer() - start))))

waveFSPS = np.load(youngAndOldSEDsPath+'waveFSPS.npy')
waveMask = (waveFSPS >= 1e3) & (waveFSPS <= 2e4)
waveFSPS = waveFSPS[waveMask]

maxIter = 2000

if fit:
	# simple TEA (bump skew = 6.95)
	TEAToCalzettiBestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	TEAToCardelliBestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	TEAToPowerLawBestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	TEAToKriekAndConroyBestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	TEAToCalzettiNoFracBestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	TEAToCardelliNoFracBestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	TEAToPowerLawNoFracBestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	TEAToKriekAndConroyNoFracBestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	# load best common models
	calzettiBestParams = np.load('bestParams/calzettiBestParams.npy')
	cardelliBestParams = np.load('bestParams/cardelliBestParams.npy')
	powerLawBestParams = np.load('bestParams/powerLawBestParams.npy')
	kriekAndConroyBestParams = np.load('bestParams/kriekAndConroyBestParams.npy')
	calzettiNoFracBestParams = np.load('bestParams/calzettiNoFracBestParams.npy')
	cardelliNoFracBestParams = np.load('bestParams/cardelliNoFracBestParams.npy')
	powerLawNoFracBestParams = np.load('bestParams/powerLawNoFracBestParams.npy')
	kriekAndConroyNoFracBestParams = np.load('bestParams/kriekAndConroyNoFracBestParams.npy')
	for i in range(len(singleNames[minAvMask])): # only loop over galaxies with minimum Av > 0.2
		galaxy = singleNames[minAvMask][i]
		print('fitting', galaxy)
		nameMaskFull = names == galaxy
		faceIndex = np.argmax(axisRatios[nameMaskFull])
		edgeIndex = np.argmin(axisRatios[nameMaskFull])
		closestAge = getClosestAgeThreshold(singleNames[minAvMask][i], ageThreshold)
		age = str(closestAge)
		youngSpec_full = np.load(youngAndOldSEDsPath+galaxy+'/'+age+'/young.npy')[waveMask]
		oldSpec_full = np.load(youngAndOldSEDsPath+galaxy+'/'+age+'/old.npy')[waveMask]
		youngSpec = matchWavelengths(waveFSPS, youngSpec_full)
		oldSpec = matchWavelengths(waveFSPS, oldSpec_full)
		youngSpecExt = matchWavelengthsExt(waveFSPS, youngSpec_full)
		oldSpecExt = matchWavelengthsExt(waveFSPS, oldSpec_full)
		# get best attenuation curves for common models and fit
		# calzetti
		attCommon = bestCalzetti(calzettiBestParams[i,0,:], 'face-on', 'frac')
		TEAToCalzettiBestParams[i,0,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		attCommon = bestCalzetti(calzettiBestParams[i,1,:], 'edge-on', 'frac')
		TEAToCalzettiBestParams[i,1,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		# cardelli
		attCommon = bestCardelli(cardelliBestParams[i,0,:], 'face-on', 'frac')
		TEAToCardelliBestParams[i,0,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		attCommon = bestCardelli(cardelliBestParams[i,1,:], 'edge-on', 'frac')
		TEAToCardelliBestParams[i,1,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		# power law
		attCommon = bestPowerLaw(powerLawBestParams[i,0,:], 'face-on', 'frac')
		TEAToPowerLawBestParams[i,0,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		attCommon = bestPowerLaw(powerLawBestParams[i,1,:], 'edge-on', 'frac')
		TEAToPowerLawBestParams[i,1,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		# kriek and conroy
		attCommon = bestKriekAndConroy(kriekAndConroyBestParams[i,0,:], 'face-on', 'frac')
		TEAToKriekAndConroyBestParams[i,0,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		attCommon = bestKriekAndConroy(kriekAndConroyBestParams[i,1,:], 'edge-on', 'frac')
		TEAToKriekAndConroyBestParams[i,1,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		# no fracNoDust
		# calzetti
		attCommon = bestCalzetti(calzettiNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		TEAToCalzettiNoFracBestParams[i,0,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		attCommon = bestCalzetti(calzettiNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		TEAToCalzettiNoFracBestParams[i,1,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		# cardelli
		attCommon = bestCardelli(cardelliNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		TEAToCardelliNoFracBestParams[i,0,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		attCommon = bestCardelli(cardelliNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		TEAToCardelliNoFracBestParams[i,1,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		# power law
		attCommon = bestPowerLaw(powerLawNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		TEAToPowerLawNoFracBestParams[i,0,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		attCommon = bestPowerLaw(powerLawNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		TEAToPowerLawNoFracBestParams[i,1,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		# kriek and conroy
		attCommon = bestKriekAndConroy(kriekAndConroyNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		TEAToKriekAndConroyNoFracBestParams[i,0,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
		attCommon = bestKriekAndConroy(kriekAndConroyNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		TEAToKriekAndConroyNoFracBestParams[i,1,:] = opt.dual_annealing(TEAToCommonMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10))).x
	os.system('mkdir -p bestTEAToCommonParams/')
	np.save('bestTEAToCommonParams/TEAToCalzettiBestParams.npy', TEAToCalzettiBestParams)
	np.save('bestTEAToCommonParams/TEAToCardelliBestParams.npy', TEAToCardelliBestParams)
	np.save('bestTEAToCommonParams/TEAToPowerLawBestParams.npy', TEAToPowerLawBestParams)
	np.save('bestTEAToCommonParams/TEAToKriekAndConroyBestParams.npy', TEAToKriekAndConroyBestParams)
	np.save('bestTEAToCommonParams/TEAToCalzettiNoFracBestParams.npy', TEAToCalzettiNoFracBestParams)
	np.save('bestTEAToCommonParams/TEAToCardelliNoFracBestParams.npy', TEAToCardelliNoFracBestParams)
	np.save('bestTEAToCommonParams/TEAToPowerLawNoFracBestParams.npy', TEAToPowerLawNoFracBestParams)
	np.save('bestTEAToCommonParams/TEAToKriekAndConroyNoFracBestParams.npy', TEAToKriekAndConroyNoFracBestParams)

if makePlots:
	fullErrors = np.zeros((len(singleNames[minAvMask]), 4, 2)) # galaxy, model, orientation
	fullErrorsNoFrac = np.zeros((len(singleNames[minAvMask]), 4, 2)) # galaxy, model, orientation
	calzettiBestParams = np.load('bestParams/calzettiBestParams.npy')
	cardelliBestParams = np.load('bestParams/cardelliBestParams.npy')
	powerLawBestParams = np.load('bestParams/powerLawBestParams.npy')
	kriekAndConroyBestParams = np.load('bestParams/kriekAndConroyBestParams.npy')
	calzettiNoFracBestParams = np.load('bestParams/calzettiNoFracBestParams.npy')
	cardelliNoFracBestParams = np.load('bestParams/cardelliNoFracBestParams.npy')
	powerLawNoFracBestParams = np.load('bestParams/powerLawNoFracBestParams.npy')
	kriekAndConroyNoFracBestParams = np.load('bestParams/kriekAndConroyNoFracBestParams.npy')
	# load TEA params
	TEAToCalzettiBestParams = np.load('bestTEAToCommonParams/TEAToCalzettiBestParams.npy')
	TEAToCardelliBestParams = np.load('bestTEAToCommonParams/TEAToCardelliBestParams.npy')
	TEAToPowerLawBestParams = np.load('bestTEAToCommonParams/TEAToPowerLawBestParams.npy')
	TEAToKriekAndConroyBestParams = np.load('bestTEAToCommonParams/TEAToKriekAndConroyBestParams.npy')
	TEAToCalzettiNoFracBestParams = np.load('bestTEAToCommonParams/TEAToCalzettiNoFracBestParams.npy')
	TEAToCardelliNoFracBestParams = np.load('bestTEAToCommonParams/TEAToCardelliNoFracBestParams.npy')
	TEAToPowerLawNoFracBestParams = np.load('bestTEAToCommonParams/TEAToPowerLawNoFracBestParams.npy')
	TEAToKriekAndConroyNoFracBestParams = np.load('bestTEAToCommonParams/TEAToKriekAndConroyNoFracBestParams.npy')
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		print('plotting', galaxy)
		nameMaskFull = names == galaxy
		faceIndex = np.argmax(axisRatios[nameMaskFull])
		edgeIndex = np.argmin(axisRatios[nameMaskFull])
		closestAge = getClosestAgeThreshold(singleNames[minAvMask][i], ageThreshold)
		age = str(closestAge)
		youngSpec_full = np.load(youngAndOldSEDsPath+galaxy+'/'+age+'/young.npy')[waveMask]
		oldSpec_full = np.load(youngAndOldSEDsPath+galaxy+'/'+age+'/old.npy')[waveMask]
		youngSpec = matchWavelengths(waveFSPS, youngSpec_full)
		oldSpec = matchWavelengths(waveFSPS, oldSpec_full)
		youngSpecExt = matchWavelengthsExt(waveFSPS, youngSpec_full)
		oldSpecExt = matchWavelengthsExt(waveFSPS, oldSpec_full)
		# calzetti
		attCommon = bestCalzetti(calzettiBestParams[i,0,:], 'face-on', 'frac')
		attTEA = bestTEA(TEAToCalzettiBestParams[i,0,:])
		fullErrors[i,0,0] = calcTEAToCommonError(attTEA)
		plotCurves('Calzetti', 'face-on', 'frac')
		attCommon = bestCalzetti(calzettiBestParams[i,1,:], 'edge-on', 'frac')
		attTEA = bestTEA(TEAToCalzettiBestParams[i,1,:])
		fullErrors[i,0,1] = calcTEAToCommonError(attTEA)
		plotCurves('Calzetti', 'edge-on', 'frac')
		# cardelli
		attCommon = bestCardelli(cardelliBestParams[i,0,:], 'face-on', 'frac')
		attTEA = bestTEA(TEAToCardelliBestParams[i,0,:])
		fullErrors[i,1,0] = calcTEAToCommonError(attTEA)
		plotCurves('Cardelli', 'face-on', 'frac')
		attCommon = bestCardelli(cardelliBestParams[i,1,:], 'edge-on', 'frac')
		attTEA = bestTEA(TEAToCardelliBestParams[i,1,:])
		fullErrors[i,1,1] = calcTEAToCommonError(attTEA)
		plotCurves('Cardelli', 'edge-on', 'frac')
		# power law
		attCommon = bestPowerLaw(powerLawBestParams[i,0,:], 'face-on', 'frac')
		attTEA = bestTEA(TEAToPowerLawBestParams[i,0,:])
		fullErrors[i,2,0] = calcTEAToCommonError(attTEA)
		plotCurves('PowerLaw', 'face-on', 'frac')
		attCommon = bestPowerLaw(powerLawBestParams[i,1,:], 'edge-on', 'frac')
		attTEA = bestTEA(TEAToPowerLawBestParams[i,1,:])
		fullErrors[i,2,1] = calcTEAToCommonError(attTEA)
		plotCurves('PowerLaw', 'edge-on', 'frac')
		# kriek and conroy
		attCommon = bestKriekAndConroy(kriekAndConroyBestParams[i,0,:], 'face-on', 'frac')
		attTEA = bestTEA(TEAToKriekAndConroyBestParams[i,0,:])
		fullErrors[i,3,0] = calcTEAToCommonError(attTEA)
		#plotCurves('KriekAndConroy', 'face-on', 'frac')
		attCommon = bestKriekAndConroy(kriekAndConroyBestParams[i,1,:], 'edge-on', 'frac')
		attTEA = bestTEA(TEAToKriekAndConroyBestParams[i,1,:])
		fullErrors[i,3,1] = calcTEAToCommonError(attTEA)
		plotCurves('KriekAndConroy', 'edge-on', 'frac')
		# no fracNoDust
		# calzetti
		attCommon = bestCalzetti(calzettiNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		attTEA = bestTEA(TEAToCalzettiNoFracBestParams[i,0,:])
		fullErrorsNoFrac[i,0,0] = calcTEAToCommonError(attTEA)
		plotCurves('Calzetti', 'face-on', 'noFrac')
		attCommon = bestCalzetti(calzettiNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		attTEA = bestTEA(TEAToCalzettiNoFracBestParams[i,1,:])
		fullErrorsNoFrac[i,0,1] = calcTEAToCommonError(attTEA)
		plotCurves('Calzetti', 'edge-on', 'noFrac')
		# cardelli
		attCommon = bestCardelli(cardelliNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		attTEA = bestTEA(TEAToCardelliNoFracBestParams[i,0,:])
		fullErrorsNoFrac[i,1,0] = calcTEAToCommonError(attTEA)
		plotCurves('Cardelli', 'face-on', 'noFrac')
		attCommon = bestCardelli(cardelliNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		attTEA = bestTEA(TEAToCardelliNoFracBestParams[i,1,:])
		fullErrorsNoFrac[i,1,1] = calcTEAToCommonError(attTEA)
		plotCurves('Cardelli', 'edge-on', 'noFrac')
		# power law
		attCommon = bestPowerLaw(powerLawNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		attTEA = bestTEA(TEAToPowerLawNoFracBestParams[i,0,:])
		fullErrorsNoFrac[i,2,0] = calcTEAToCommonError(attTEA)
		plotCurves('PowerLaw', 'face-on', 'noFrac')
		attCommon = bestPowerLaw(powerLawNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		attTEA = bestTEA(TEAToPowerLawNoFracBestParams[i,1,:])
		fullErrorsNoFrac[i,2,1] = calcTEAToCommonError(attTEA)
		plotCurves('PowerLaw', 'edge-on', 'noFrac')
		# kriek and conroy
		attCommon = bestKriekAndConroy(kriekAndConroyNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		attTEA = bestTEA(TEAToKriekAndConroyNoFracBestParams[i,0,:])
		fullErrorsNoFrac[i,3,0] = calcTEAToCommonError(attTEA)
		plotCurves('KriekAndConroy', 'face-on', 'noFrac')
		attCommon = bestKriekAndConroy(kriekAndConroyNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		attTEA = bestTEA(TEAToKriekAndConroyNoFracBestParams[i,1,:])
		fullErrorsNoFrac[i,3,1] = calcTEAToCommonError(attTEA)
		plotCurves('KriekAndConroy', 'edge-on', 'noFrac')
	plotErrorsSeparate(fullErrors, 'frac')
	plotErrorsSeparate(fullErrorsNoFrac, 'noFrac')


