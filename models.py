from math import log10, log

class FreeSpace:

	def __init__(self, freq = 800):

		self.freq = freq

	# end

	def pathLoss(self, dist):

		return 32.44 + 20*log10(dist) + 20*log10(self.freq)
		
	# end

# end


class OkumuraHata:

	def __init__(self, freq = 800, txH = 90, rxH = 1.2, cityKind = 'medium', areaKind = 'urban', checkFreqRange = True):

		self.freq = freq
		self.txH = txH
		self.rxH = rxH
		self.cityKind = cityKind
		self.areaKind = areaKind
		self.checkFreqRange = checkFreqRange

		self.invalidFreqRange = (freq < 500) or (freq > 1500)

	# end

	def pathLoss(self, dist):

		if self.checkFreqRange and self.invalidFreqRange:
			print("OkumuraHata frequency range is 500MHz~1500MHz")
			return
		# end

		if (self.freq <= 200) and (self.cityKind == 'large'):
			a = 8.29*(log10(1.54*self.rxH))^2 - 1.1
		elif (self.freq >= 400) and (self.cityKind == 'large'):
			a = 3.2*((log10(11.75*self.rxH)^2)) - 4.97
		else:
			a = (1.1*log10(self.freq-0.7))*self.rxH - (1.56*log10(self.freq-0.8))

		loss = 69.55 + (26.16)*log10(self.freq)-13.82*log10(self.txH) - a + (44.9-6.55*log10(self.txH))*log10(dist)

		if (self.areaKind == 'open'):
			loss -= 4.78*((log10(self.freq))^2) + 18.33*log10(self.freq) - 40.94
		elif (self.areaKind == 'suburban'):
			loss -= 2*(log10(self.freq/28.0))^2 - 5.4

		return loss

	# end

# end

class Cost231Hata:

	def __init__(self, freq, txH, rxH, areaKind, checkFreqRange):

		self.freq = freq
		self.txH = txH
		self.rxH = rxH
		self.areaKind = areaKind
		self.checkFreqRange = checkFreqRange

		self.invalidFreqRange = (freq < 1500) or (freq > 2000)

	# end

	def pathLoss(self, dist):

		if self.checkFreqRange and self.invalidFreqRange:
			print("Cost231Hata frequency range is 1500MHz~2000MHz")
			return
		# end

		ar = (1.1*log10(self.freq)-0.7)*self.rxH - (1.56*log(self.freq)-0.8)
		loss = 46.3 + 33.9*log10(self.freq) - 13.82*log10(self.txH) - ar + (44.9-6.55*log(self.txH))*log(dist)

		if (self.areaKind == 'urban'):
			loss += 3
		# end

		return loss

	# end

# end

class Cost231:

	def __init__(self, freq, txH, rxH, ws, bs, hr, cityKind, checkFreqRange):

		self.freq = freq
		self.txH = txH
		self.rxH = rxH
		self.ws = ws
		self.bs = bs
		self.hr = hr
		self.cityKind = cityKind
		self.checkFreqRange = checkFreqRange

		self.invalidFreqRange = (freq < 150) or (freq > 2000)

	# end

	def pathLoss(self, dist):

		deltaH = self.rxH / self.txH
		Lbsh = 0 if (self.hr > self.txH) else 18*log(1+deltaH)
		Ka = 54
		Kd = 18
		Kf = 4

		if (self.txH <= self.hr) and (dist >= 0.5):
			Ka -= 0.8*deltaH if (dist >= 0.5) else 0.8*deltaH*(dist/0.5)
		# end

		if (self.txH < self.hr):
			Kd -= 15*(self.txH-self.hr)/(self.hr-self.rxH)
		# end

		Kf += 0.7*(self.freq/925-1) if (self.cityKind == 'small') else 1.5*(self.freq/925-1)

		Lo = 32.4+20*log10(dist)+20*log10(self.freq)
		Lrts = 8.2+10*log(self.ws) + 10*log10(self.freq) + 10*log(deltaH)
		Lmsd = Lbsh + Ka + Kd*log10(dist) + Kf*log(self.freq) - 9*log10(self.bs)

		return Lo + Lrts + Lmsd

	# end

# end

