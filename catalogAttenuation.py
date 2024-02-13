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
	xbins /= 1e9 # convert to Gyrs
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
	plt.plot(np.log10(errorWave), attenuationMags, label='Calzetti', alpha=0.5, linewidth=4.5, linestyle='dashed')
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=4.5)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=42)
	plt.ylabel(r'$A_{\lambda}$',fontsize=42)
	plt.xticks(fontsize=42)
	plt.yticks(fontsize=42)
	plt.legend(fontsize=30)
	plt.savefig(plotPath+'best'+name+'/'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

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
	plt.plot(np.log10(errorWave), attenuationMags, label='Cardelli', alpha=0.5, linewidth=4.5, linestyle='dashed')
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=4.5)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=42)
	plt.ylabel(r'$A_{\lambda}$',fontsize=42)
	plt.xticks(fontsize=42)
	plt.yticks(fontsize=42)
	plt.legend(fontsize=30)
	plt.savefig(plotPath+'best'+name+'/'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

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
	plt.plot(np.log10(errorWave), attenuationMags, label='Power Law', alpha=0.5, linewidth=4.5, linestyle='dashed')
	#plt.plot(np.log10(attenuation_wave), attenuation_mags[nameMaskFull][faceIndex], label='Catalog', alpha=0.5)
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=4.5)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=42)
	plt.ylabel(r'$A_{\lambda}$',fontsize=42)
	plt.xticks(fontsize=42)
	plt.yticks(fontsize=42)
	plt.legend(fontsize=30)
	plt.savefig(plotPath+'best'+name+'/'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

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
	plt.plot(np.log10(errorWave), attenuationMags, label='Kriek and Conroy', alpha=0.5, linewidth=4.5, linestyle='dashed')
	#plt.plot(np.log10(attenuation_wave), attenuation_mags[nameMaskFull][faceIndex], label='Catalog', alpha=0.5)
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=4.5)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=42)
	plt.ylabel(r'$A_{\lambda}$',fontsize=42)
	plt.xticks(fontsize=42)
	plt.yticks(fontsize=42)
	plt.legend(fontsize=30)
	plt.savefig(plotPath+'best'+name+'/'+name+'_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

# old version with 4 free parameters
#def TEAMin(x):
#	Av = x[0]
#	dustIndex = x[1]
#	bumpStrength = x[2]
#	bumpSkew = x[3]
#	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength, bumpSkew)
#	return calcError(attenuationMags)
#
#def plotTEA(x, orientation):
#	if orientation == 'face-on':
#		index = faceIndex
#	elif orientation == 'edge-on':
#		index = edgeIndex
#	Av = x[0]
#	dustIndex = x[1]
#	bumpStrength = x[2]
#	bumpSkew = x[3]
#	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength, bumpSkew)
#	os.system('mkdir -p '+plotPath+'bestTEA/')
#	plt.figure(figsize=(10,8))
#	plt.plot(np.log10(errorWave), attenuationMags, label='TEA', alpha=0.5, linewidth=4.5, linestyle='dashed')
#	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=4.5)
#	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=42)
#	plt.ylabel(r'$A_{\lambda}$',fontsize=42)
#	plt.xticks(fontsize=42)
#	plt.yticks(fontsize=42)
#	plt.legend(fontsize=30)
#	plt.savefig(plotPath+'bestTEA/TEA_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
#	plt.close()

def TEAMin(x):
	Av = x[0]
	dustIndex = x[1]
	bumpStrength = x[2]
	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength)
	return calcError(attenuationMags)

def plotTEA(x, orientation):
	if orientation == 'face-on':
		index = faceIndex
	elif orientation == 'edge-on':
		index = edgeIndex
	Av = x[0]
	dustIndex = x[1]
	bumpStrength = x[2]
	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, bumpStrength)
	os.system('mkdir -p '+plotPath+'bestTEA/')
	plt.figure(figsize=(10,8))
	plt.plot(np.log10(errorWave), attenuationMags, label='TEA', alpha=0.5, linewidth=4.5, linestyle='dashed')
	plt.plot(np.log10(errorWave), matchedAttCatalog[nameMaskFull][index], label='Catalog', alpha=0.5, linewidth=4.5)
	plt.xlabel(r'$\log_{10}(\lambda \, / \, \AA)$', fontsize=42)
	plt.ylabel(r'$A_{\lambda}$',fontsize=42)
	plt.xticks(fontsize=42)
	plt.yticks(fontsize=42)
	plt.legend(fontsize=30)
	plt.savefig(plotPath+'bestTEA/simpleTEA_'+galaxy+'_'+orientation+'_attenuation.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def plotErrorsSeparate(fullErrors, frac):
	if frac == 'noFrac':
		folder = 'residualsNoFrac/'
		ext = 'NoFrac'
	else:
		folder = 'residuals/'
		ext = ''
	size = 200
	plt.figure(figsize=(10,8))
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,0,0]), label='Calzetti', color='pink', marker='s', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,1,0]), label='Cardelli', color='sandybrown', marker='^', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,2,0]), label='Power Law', color='tomato', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,3,0]), label='Kriek and Conroy', color='cornflowerblue', marker='D', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,5,0]), label='TEA', color='darkgreen', marker='*', alpha=0.7, s=size*1.5, linewidth=0)
	#plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,5,0]), label='TEATwoComponent', color='maroon', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.xlabel(r'$\log_{10}(Stellar \; Mass \, / \, M_{\odot})$', fontsize=28)
	plt.ylabel(r'$\log_{10}(RMSE)$',fontsize=28)
	plt.title('Face-on',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+folder+'face-on_errors'+ext+'.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()
	plt.figure(figsize=(10,8))
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,0,1]), label='Calzetti', color='pink', marker='s', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,1,1]), label='Cardelli', color='sandybrown', marker='^', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,2,1]), label='Power Law', color='tomato', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,3,1]), label='Kriek and Conroy', color='cornflowerblue', marker='D', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,5,1]), label='TEA', color='darkgreen', marker='*', alpha=0.7, s=size*1.5, linewidth=0)
	#plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,5,1]), label='TEATwoComponent', color='maroon', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.xlabel(r'$\log_{10}(Stellar \; Mass \, / \, M_{\odot})$', fontsize=28)
	plt.ylabel(r'$\log_{10}(RMSE)$',fontsize=28)
	plt.title('Edge-on',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+folder+'edge-on_errors'+ext+'.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def plotErrorsSeparateShifted(fullErrors, frac):
	if frac == 'noFrac':
		folder = 'residualsNoFrac/'
		ext = 'NoFrac'
	else:
		folder = 'residuals/'
		ext = ''
	size = 200
	plt.figure(figsize=(10,8))
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,0,0]/fullErrors[:,5,0]), label='Calzetti', color='pink', marker='s', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,1,0]/fullErrors[:,5,0]), label='Cardelli', color='sandybrown', marker='^', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,2,0]/fullErrors[:,5,0]), label='Power Law', color='tomato', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,3,0]/fullErrors[:,5,0]), label='Kriek and Conroy', color='cornflowerblue', marker='D', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,5,0]/fullErrors[:,5,0]), label='TEA', color='darkgreen', marker='*', alpha=0.7, s=size*1.5, linewidth=0)
	#plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,5,0]), label='TEATwoComponent', color='maroon', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.xlabel(r'$\log_{10}(Stellar \; Mass \, / \, M_{\odot})$', fontsize=28)
	plt.ylabel(r'$\log_{10}(RMSE \, / \, RMSE_{TEA})$',fontsize=28)
	plt.title('Face-on',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+folder+'shifted_face-on_errors'+ext+'.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()
	plt.figure(figsize=(10,8))
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,0,1]/fullErrors[:,5,1]), label='Calzetti', color='pink', marker='s', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,1,1]/fullErrors[:,5,1]), label='Cardelli', color='sandybrown', marker='^', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,2,1]/fullErrors[:,5,1]), label='Power Law', color='tomato', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,3,1]/fullErrors[:,5,1]), label='Kriek and Conroy', color='cornflowerblue', marker='D', alpha=0.7, s=size, linewidth=0)
	plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,5,1]/fullErrors[:,5,1]), label='TEA', color='darkgreen', marker='*', alpha=0.7, s=size*1.5, linewidth=0)
	#plt.scatter(np.log10(singleStellarMass[minAvMask]), np.log10(fullErrors[:,5,1]), label='TEATwoComponent', color='maroon', marker='o', alpha=0.7, s=size, linewidth=0)
	plt.xlabel(r'$\log_{10}(Stellar \; Mass \, / \, M_{\odot})$', fontsize=28)
	plt.ylabel(r'$\log_{10}(RMSE \, / \, RMSE_{TEA})$',fontsize=28)
	plt.title('Edge-on',fontsize=28)
	plt.xticks(fontsize=28)
	plt.yticks(fontsize=28)
	plt.legend(fontsize=20)
	plt.savefig(plotPath+folder+'shifted_edge-on_errors'+ext+'.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
	plt.close()

def calcAllErrors(calzettiBestParams, cardelliBestParams, powerLawBestParams, kriekAndConroyBestParams, TEABestParams, frac):
	if frac == 'noFrac':
		folder = 'residualsNoFrac/'
	else:
		folder = 'residuals/'
	errors = np.zeros((5,2)) # model, orientation
	os.system('mkdir -p '+plotPath+folder)
	# calzetti face-on
	Av = calzettiBestParams[0,0]
	fracNoDust = calzettiBestParams[0,1]
	dustySpec, attenuationMags = dm.calzetti(errorWave, youngSpec+oldSpec, Av, fracNoDust)
	errors[0,0] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][faceIndex])**2), axis=-1) / len(errorWave)
	# calzetti edge-on
	Av = calzettiBestParams[1,0]
	fracNoDust = calzettiBestParams[1,1]
	dustySpec, attenuationMags = dm.calzetti(errorWave, youngSpec+oldSpec, Av, fracNoDust)
	errors[0,1] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][edgeIndex])**2), axis=-1) / len(errorWave)
	# cardelli face-on
	Av = cardelliBestParams[0,0]
	mwr = cardelliBestParams[0,1]
	uvb = cardelliBestParams[0,2]
	AvYoung = cardelliBestParams[0,3]
	dustIndexYoung = cardelliBestParams[0,4]
	fracNoDust = cardelliBestParams[0,5]
	fracNoDustYoung = cardelliBestParams[0,6]
	dustySpec, attenuationMags = dm.cardelli(errorWaveExt, youngSpecExt, oldSpecExt, Av, mwr, uvb, AvYoung, 
			 					 dustIndexYoung, fracNoDust, fracNoDustYoung)
	dustySpec = dustySpec[:-1]
	attenuationMags = attenuationMags[:-1]
	errors[1,0] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][faceIndex])**2), axis=-1) / len(errorWave)
	# cardelli edge-on
	Av = cardelliBestParams[1,0]
	mwr = cardelliBestParams[1,1]
	uvb = cardelliBestParams[1,2]
	AvYoung = cardelliBestParams[1,3]
	dustIndexYoung = cardelliBestParams[1,4]
	fracNoDust = cardelliBestParams[1,5]
	fracNoDustYoung = cardelliBestParams[1,6]
	dustySpec, attenuationMags = dm.cardelli(errorWaveExt, youngSpecExt, oldSpecExt, Av, mwr, uvb, AvYoung, 
			 					 dustIndexYoung, fracNoDust, fracNoDustYoung)
	dustySpec = dustySpec[:-1]
	attenuationMags = attenuationMags[:-1]
	errors[1,1] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][edgeIndex])**2), axis=-1) / len(errorWave)
	# power law face-on
	Av = powerLawBestParams[0,0]
	dustIndex = powerLawBestParams[0,1]
	AvYoung = powerLawBestParams[0,2]
	dustIndexYoung = powerLawBestParams[0,3]
	fracNoDust = powerLawBestParams[0,4]
	fracNoDustYoung = powerLawBestParams[0,5]
	dustySpec, attenuationMags = dm.powerLaw(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	errors[2,0] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][faceIndex])**2), axis=-1) / len(errorWave)
	# power law edge-on
	Av = powerLawBestParams[1,0]
	dustIndex = powerLawBestParams[1,1]
	AvYoung = powerLawBestParams[1,2]
	dustIndexYoung = powerLawBestParams[1,3]
	fracNoDust = powerLawBestParams[1,4]
	fracNoDustYoung = powerLawBestParams[1,5]
	dustySpec, attenuationMags = dm.powerLaw(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	errors[2,1] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][edgeIndex])**2), axis=-1) / len(errorWave)
	# kriek and conroy face-on
	Av = kriekAndConroyBestParams[0,0]
	dustIndex = kriekAndConroyBestParams[0,1]
	AvYoung = kriekAndConroyBestParams[0,2]
	dustIndexYoung = kriekAndConroyBestParams[0,3]
	fracNoDust = kriekAndConroyBestParams[0,4]
	fracNoDustYoung = kriekAndConroyBestParams[0,5]
	dustySpec, attenuationMags = dm.kriekAndConroy(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	errors[3,0] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][faceIndex])**2), axis=-1) / len(errorWave)
	# kriek and conroy edge-on
	Av = kriekAndConroyBestParams[1,0]
	dustIndex = kriekAndConroyBestParams[1,1]
	AvYoung = kriekAndConroyBestParams[1,2]
	dustIndexYoung = kriekAndConroyBestParams[1,3]
	fracNoDust = kriekAndConroyBestParams[1,4]
	fracNoDustYoung = kriekAndConroyBestParams[1,5]
	dustySpec, attenuationMags = dm.kriekAndConroy(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 AvYoung, dustIndexYoung, fracNoDust, fracNoDustYoung)
	errors[3,1] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][edgeIndex])**2), axis=-1) / len(errorWave)
	# TEA face-on
	Av = TEABestParams[0,0]
	dustIndex = TEABestParams[0,1]
	bumpStrength = TEABestParams[0,2]
	bumpSkew = TEABestParams[0,3]
	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 bumpStrength, bumpSkew)
	errors[4,0] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][faceIndex])**2), axis=-1) / len(errorWave)
	# TEA edge-on
	Av = TEABestParams[1,0]
	dustIndex = TEABestParams[1,1]
	bumpStrength = TEABestParams[1,2]
	bumpSkew = TEABestParams[1,3]
	dustySpec, attenuationMags = dm.TEA(errorWave, youngSpec, oldSpec, Av, dustIndex, 
								 bumpStrength, bumpSkew)
	errors[4,1] = np.sum(np.sqrt((attenuationMags - matchedAttCatalog[nameMaskFull][edgeIndex])**2), axis=-1) / len(errorWave)
	return errors

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

# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #

storeSSPs = False
fit = False
makePlots = True

youngAndOldSEDsPath = 'youngAndOldSEDs/'

plotPath = 'Plots/'
os.system('mkdir -p '+plotPath)

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

if fit:
	maxIter = 2000
	calzettiBestParams = np.zeros((len(singleNames[minAvMask]), 2, 2)) # galaxy, orientation, parameter
	cardelliBestParams = np.zeros((len(singleNames[minAvMask]), 2, 7))
	powerLawBestParams = np.zeros((len(singleNames[minAvMask]), 2, 6))
	kriekAndConroyBestParams = np.zeros((len(singleNames[minAvMask]), 2, 6))
	TEABestParams = np.zeros((len(singleNames[minAvMask]), 2, 4))
	simpleTEABestParams = np.zeros((len(singleNames[minAvMask]), 2, 3))
	# fracNoDust and fracNoDustYoung fixed to 0
	calzettiNoFracBestParams = np.zeros((len(singleNames[minAvMask]), 2, 2)) # galaxy, orientation, parameter
	cardelliNoFracBestParams = np.zeros((len(singleNames[minAvMask]), 2, 7))
	powerLawNoFracBestParams = np.zeros((len(singleNames[minAvMask]), 2, 6))
	kriekAndConroyNoFracBestParams = np.zeros((len(singleNames[minAvMask]), 2, 6))
	for i in range(len(singleNames[minAvMask])): 
		galaxy = singleNames[minAvMask][i]
		print('fitting', galaxy)
		nameMaskFull = names == galaxy
		faceIndex = np.argmax(axisRatios[nameMaskFull])
		edgeIndex = np.argmin(axisRatios[nameMaskFull])
		trueAv = Av[nameMaskFull][faceIndex]
		closestAge = getClosestAgeThreshold(singleNames[minAvMask][i], ageThreshold)
		age = str(closestAge)
		youngSpec_full = np.load(youngAndOldSEDsPath+galaxy+'/'+age+'/young.npy')[waveMask]
		oldSpec_full = np.load(youngAndOldSEDsPath+galaxy+'/'+age+'/old.npy')[waveMask]
		youngSpec = matchWavelengths(waveFSPS, youngSpec_full)
		oldSpec = matchWavelengths(waveFSPS, oldSpec_full)
		youngSpecExt = matchWavelengthsExt(waveFSPS, youngSpec_full)
		oldSpecExt = matchWavelengthsExt(waveFSPS, oldSpec_full)
		# find best parameters
		# face-on
		currentMatchedAttCatalog = matchedAttCatalog[nameMaskFull][faceIndex]
		calzettiResultFace = opt.dual_annealing(calzettiMin, maxiter=maxIter,
						 bounds=((0.,10.), (0., 1.0)))
		cardelliResultFace = opt.dual_annealing(cardelliMin, maxiter=maxIter,
						 bounds=((0.,10.), (0., 10.0), (0., 10.0), (0., 10.0), (-5., 0.), (0., 1.0), (0., 1.0)))
		powerLawResultFace = opt.dual_annealing(powerLawMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.,10.), (-5., 0.), (0., 1.), (0., 1.)))
		kriekAndConroyResultFace = opt.dual_annealing(kriekAndConroyMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.,10.), (-5., 0.), (0., 1.), (0., 1.)))
		TEAResultFace = opt.dual_annealing(TEAMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10)))
		# fracNoDust and fracNoDustYoung fixed to 0
		calzettiNoFracResultFace = opt.minimize_scalar(calzettiMinNoFrac,
						 bounds=((0.,10.)))
		cardelliNoFracResultFace = opt.dual_annealing(cardelliMinNoFrac, maxiter=maxIter,
						 bounds=((0.,10.), (0., 10.0), (0., 10.0), (0., 10.0), (-5., 0.)))
		powerLawNoFracResultFace = opt.dual_annealing(powerLawMinNoFrac, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.,10.), (-5., 0.)))
		kriekAndConroyNoFracResultFace = opt.dual_annealing(kriekAndConroyMinNoFrac, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.,10.), (-5., 0.)))
		# edge-on
		currentMatchedAttCatalog = matchedAttCatalog[nameMaskFull][edgeIndex]
		calzettiResultEdge = opt.dual_annealing(calzettiMin, maxiter=maxIter,
						 bounds=((0.,10.), (0., 1.0)))
		cardelliResultEdge = opt.dual_annealing(cardelliMin, maxiter=maxIter,
						 bounds=((0.,10.), (0., 10.0), (0., 10.0), (0., 10.0), (-5., 0.), (0., 1.0), (0., 1.0)))
		powerLawResultEdge = opt.dual_annealing(powerLawMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.,10.), (-5., 0.), (0., 1.), (0., 1.)))
		kriekAndConroyResultEdge = opt.dual_annealing(kriekAndConroyMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.,10.), (-5., 0.), (0., 1.), (0., 1.)))
		TEAResultEdge = opt.dual_annealing(TEAMin, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.0,10)))
		# fracNoDust and fracNoDustYoung fixed to 0
		calzettiNoFracResultEdge = opt.minimize_scalar(calzettiMinNoFrac,
						 bounds=((0.,10.)))
		cardelliNoFracResultEdge = opt.dual_annealing(cardelliMinNoFrac, maxiter=maxIter,
						 bounds=((0.,10.), (0., 10.0), (0., 10.0), (0., 10.0), (-5., 0.)))
		powerLawNoFracResultEdge = opt.dual_annealing(powerLawMinNoFrac, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.,10.), (-5., 0.)))
		kriekAndConroyNoFracResultEdge = opt.dual_annealing(kriekAndConroyMinNoFrac, maxiter=maxIter,
						 bounds=((0.,10.), (-5., 0.), (0.,10.), (-5., 0.)))
		# face-on
		calzettiBestParams[i,0,:] = calzettiResultFace.x
		cardelliBestParams[i,0,:] = cardelliResultFace.x
		powerLawBestParams[i,0,:] = powerLawResultFace.x
		kriekAndConroyBestParams[i,0,:] = kriekAndConroyResultFace.x
		TEABestParams[i,0,:] = TEAResultFace.x
		calzettiNoFracBestParams[i,0,:] = np.append(calzettiNoFracResultFace.x, [0])
		cardelliNoFracBestParams[i,0,:] = np.append(cardelliNoFracResultFace.x, [0,0])
		powerLawNoFracBestParams[i,0,:] = np.append(powerLawNoFracResultFace.x, [0,0])
		kriekAndConroyNoFracBestParams[i,0,:] = np.append(kriekAndConroyNoFracResultFace.x, [0,0])
		# edge-on
		calzettiBestParams[i,1,:] = calzettiResultEdge.x
		cardelliBestParams[i,1,:] = cardelliResultEdge.x
		powerLawBestParams[i,1,:] = powerLawResultEdge.x
		kriekAndConroyBestParams[i,1,:] = kriekAndConroyResultEdge.x
		TEABestParams[i,1,:] = TEAResultEdge.x
		calzettiNoFracBestParams[i,1,:] = np.append(calzettiNoFracResultEdge.x, [0])
		cardelliNoFracBestParams[i,1,:] = np.append(cardelliNoFracResultEdge.x, [0,0])
		powerLawNoFracBestParams[i,1,:] = np.append(powerLawNoFracResultEdge.x, [0,0])
		kriekAndConroyNoFracBestParams[i,1,:] = np.append(kriekAndConroyNoFracResultEdge.x, [0,0])
	os.system('mkdir -p bestParams')
	np.save('bestParams/calzettiBestParams.npy', calzettiBestParams)
	np.save('bestParams/cardelliBestParams.npy', cardelliBestParams)
	np.save('bestParams/powerLawBestParams.npy', powerLawBestParams)
	np.save('bestParams/kriekAndConroyBestParams.npy', kriekAndConroyBestParams)
	np.save('bestParams/TEABestParams.npy', TEABestParams)
	np.save('bestParams/calzettiNoFracBestParams.npy', calzettiNoFracBestParams)
	np.save('bestParams/cardelliNoFracBestParams.npy', cardelliNoFracBestParams)
	np.save('bestParams/powerLawNoFracBestParams.npy', powerLawNoFracBestParams)
	np.save('bestParams/kriekAndConroyNoFracBestParams.npy', kriekAndConroyNoFracBestParams)

if makePlots:
	fullErrors = np.zeros((len(singleNames[minAvMask]), 6, 2)) # galaxy, model, orientation
	fullErrorsNoFrac = np.zeros((len(singleNames[minAvMask]), 6, 2)) # galaxy, model, orientation
	faceEdgeAv = np.zeros((len(singleNames[minAvMask]), 2)) # galaxy, orientation 
	calzettiBestParams = np.load('bestParams/calzettiBestParams.npy')
	cardelliBestParams = np.load('bestParams/cardelliBestParams.npy')
	powerLawBestParams = np.load('bestParams/powerLawBestParams.npy')
	kriekAndConroyBestParams = np.load('bestParams/kriekAndConroyBestParams.npy')
	TEABestParams = np.load('bestParams/TEABestParams.npy')
	calzettiNoFracBestParams = np.load('bestParams/calzettiNoFracBestParams.npy')
	cardelliNoFracBestParams = np.load('bestParams/cardelliNoFracBestParams.npy')
	powerLawNoFracBestParams = np.load('bestParams/powerLawNoFracBestParams.npy')
	kriekAndConroyNoFracBestParams = np.load('bestParams/kriekAndConroyNoFracBestParams.npy')
	avgAlambda = np.zeros((len(singleNames[minAvMask]), 2)) # galaxy, orientation
	closestAgeDist = np.zeros((len(singleNames[minAvMask])))
	for i in range(len(singleNames[minAvMask])):
		galaxy = singleNames[minAvMask][i]
		print('plotting', galaxy)
		nameMaskFull = names == galaxy
		faceIndex = np.argmax(axisRatios[nameMaskFull])
		edgeIndex = np.argmin(axisRatios[nameMaskFull])
		avgAlambda[i, 0] = np.average(matchedAttCatalog[nameMaskFull][faceIndex])
		avgAlambda[i, 1] = np.average(matchedAttCatalog[nameMaskFull][edgeIndex])
		faceEdgeAv[i, 0] = Av[nameMaskFull][faceIndex]
		faceEdgeAv[i, 1] = Av[nameMaskFull][edgeIndex]
		closestAge = getClosestAgeThreshold(singleNames[minAvMask][i], ageThreshold)
		closestAgeDist[i] = closestAge
		age = str(closestAge)
		youngSpec_full = np.load(youngAndOldSEDsPath+galaxy+'/'+age+'/young.npy')[waveMask]
		oldSpec_full = np.load(youngAndOldSEDsPath+galaxy+'/'+age+'/old.npy')[waveMask]
		youngSpec = matchWavelengths(waveFSPS, youngSpec_full)
		oldSpec = matchWavelengths(waveFSPS, oldSpec_full)
		youngSpecExt = matchWavelengthsExt(waveFSPS, youngSpec_full)
		oldSpecExt = matchWavelengthsExt(waveFSPS, oldSpec_full)
		plotCalzetti(calzettiBestParams[i,0,:], 'face-on', 'frac')
		plotCardelli(cardelliBestParams[i,0,:], 'face-on', 'frac')
		plotPowerLaw(powerLawBestParams[i,0,:], 'face-on', 'frac')
		plotKriekAndConroy(kriekAndConroyBestParams[i,0,:], 'face-on', 'frac')
		plotTEA(TEABestParams[i,0,:], 'face-on')
		plotCalzetti(calzettiNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		plotCardelli(cardelliNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		plotPowerLaw(powerLawNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		plotKriekAndConroy(kriekAndConroyNoFracBestParams[i,0,:], 'face-on', 'noFrac')
		plotCalzetti(calzettiBestParams[i,1,:], 'edge-on', 'frac')
		plotCardelli(cardelliBestParams[i,1,:], 'edge-on', 'frac')
		plotPowerLaw(powerLawBestParams[i,1,:], 'edge-on', 'frac')
		plotKriekAndConroy(kriekAndConroyBestParams[i,1,:], 'edge-on', 'frac')
		plotTEA(TEABestParams[i,1,:], 'edge-on')
		plotCalzetti(calzettiNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		plotCardelli(cardelliNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		plotPowerLaw(powerLawNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		plotKriekAndConroy(kriekAndConroyNoFracBestParams[i,1,:], 'edge-on', 'noFrac')
		errors = calcAllErrors(calzettiBestParams[i,:,:], cardelliBestParams[i,:,:], 
							   powerLawBestParams[i,:,:], kriekAndConroyBestParams[i,:,:],
							   TEABestParams[i,:,:], 'frac')
		errorsNoFrac = calcAllErrors(calzettiNoFracBestParams[i,:,:], cardelliNoFracBestParams[i,:,:], 
							   powerLawNoFracBestParams[i,:,:], kriekAndConroyNoFracBestParams[i,:,:],
							   TEABestParams[i,:,:], 'noFrac')
		fullErrors[i, :, :] = errors
		fullErrorsNoFrac[i, :, :] = errorsNoFrac
	plotErrorsSeparate(fullErrors, 'frac')
	plotErrorsSeparate(fullErrorsNoFrac, 'noFrac')
	plotErrorsSeparateShifted(fullErrors, 'frac')
	plotErrorsSeparateShifted(fullErrorsNoFrac, 'noFrac')
	#print('average threshold age difference from 10 Myrs in years:', np.mean(closestAgeDist*1e9 - 1e7))


