#!/usr/bin/python3

import random
import simpleaudio as sa
import numpy as np
from time import sleep, time
from numpy.random import normal
from os.path import isfile
from sys import argv

DEFAULT_SAM_RATE_HZ = 11025
EXEC_FREQ_HZ = 500
DEFAULT_TONE_FREQ_HZ = 600
DEFAULT_FADE_PCT = 5
ASSUME_AVG_WORDLEN_UNITS = 40 # how long do we assume the average word is, in units?
DEFAULT_MAX_INTENSITY_PCT = 75
DEFAULT_KEY_WPM = 20
DEFAULT_FARN_WPM = 20
DEFAULT_AVG_WORDLEN = 5 # characters
DEFAULT_WORDLEN_SIG = 2 # characters (stdev)

ALLOWED_AUDIO_SAM_RATES = {
8000,
11025,
16000,
22050,
24000,
32000,
44100,
48000,
88200,
96000,
192000
}

MORSE_ENCS = {
'A':'.-',
'B':'-...',
'C':'-.-.',
'D':'-..',
'E':'.',
'F':'..-.',
'G':'--.',
'H':'....',
'I':'..',
'J':'.---',
'K':'-.-',
'L':'.-..',
'M':'--',
'N':'-.',
'O':'---',
'P':'.--.',
'Q':'--.-',
'R':'.-.',
'S':'...',
'T':'-',
'U':'..-',
'V':'...-',
'W':'.--',
'X':'-..-',
'Y':'-.--',
'Z':'--..',
'0':'-----',
'1':'.----',
'2':'..---',
'3':'...--',
'4':'....-',
'5':'.....',
'6':'-....',
'7':'--...',
'8':'---..',
'9':'----.',
'.':'.-.-.-',
',':'--..--',
':':'---...',
'?':'..--..',
'/':'-..-.',
'\'':'.----.',
'"':'.-..-.',
'(':'-.--.',
')':'-.--.-',
'@':'.--.-.',
'+':'.-.-.',
'-':'-....-',
'=':'-...-',
' ':'/'#not technically a thing, just putting it here so we know it's supposed to be there
}

# for each symbol in the morse alphabet, how many units does it correspond to, and is it on or off?
OUTPUTS = {
'.':(1,True),
'-':(3,True),
' ':(3,False),
'/':(7,False)
}

def tryparse(num_str):
	try:
		x = float(num_str)
		return x
	except ValueError:
		print("ERROR: unable to parse numerical value: ", num_str)
		return None

def cycle_delay(start_time):
	"""
	given the start time of the cycle (this function will return the current time to facilitate in tracking this), delay the remaining cycle time
	"""
	
	cycle_len_s = 1/EXEC_FREQ_HZ
	nom_end_time = start_time + cycle_len_s
	current_time = time()
	remaining_time = max(0,nom_end_time - current_time)
	
	# delay the remaining time
	sleep(remaining_time)
	
	# return the time at loop end
	return time()

def gen_sinewave(freq_hz, sam_len_sec, amp_pct=DEFAULT_MAX_INTENSITY_PCT, sam_rate_hz=DEFAULT_SAM_RATE_HZ, fade_pct=DEFAULT_FADE_PCT):
	"""
	generate a sine wave at the specified frequency (Hz), sample rate (Hz), amplitude (number), and length (s)
	
	fade_pct is how much of the wave length will be fading in at the beginning and out at the end (so only 100-2*fade-pct % of the time will the sound be at full intensity)
	"""
	
	# max amplitude is 32768 = 2^16
	amp = (amp_pct / 100) * 32768
	
	# frequency in (angular freq) = 2pi * frequency in Hz
	ang_freq = freq_hz*2*np.pi
	
	# sine wave eq: f(x) = A * sin( om * x )
	# where A = amplitude, om = angular freq
	
	# first generate all of the x-values
	xvals = np.arange(0, sam_len_sec, 1/sam_rate_hz)
	
	# now mult by angular frequency to get the input to sin
	sin_inp = xvals * ang_freq
	
	# feed into sin
	sin_out_unscaled = np.sin(sin_inp)
	
	# scale
	sin_out_scaled = amp*sin_out_unscaled
	
	# apply fades
	fade_samples = int(len(sin_out_scaled)) * (fade_pct / 100)
	prefade = np.array([min(fade_samples, x) for x in range(len(sin_out_scaled))]) / fade_samples
	postfade = np.flip(prefade)
	out = np.multiply(sin_out_scaled,np.multiply(prefade,postfade))
	
	# convert to the proper data type
	return np.array(out, dtype=np.int16)

def text_to_morse(unencoded_str):
	"""
	encode the given string to the following characters:
		. = "dit" (one unit on)
		- = "dah" (three units on)
		  = char space (one unit off)
		/ = word space (aka ' ') (three units off)
	"""
	
	unencoded_str = unencoded_str.upper()
	
	enc_str = []
	for i,ch in enumerate(unencoded_str):
		if ch not in MORSE_ENCS:
			print("ERROR, character not known to the encoder: ",ch)
			return None
		else:
			enc_str.append(MORSE_ENCS[ch])
			if i != len(unencoded_str) - 1:
				enc_str.append(' ')
	
	return ''.join(enc_str)

def gen_rand_fake_word(avg_wordlen, wordlen_sig, available_chars):
	"""
	generate a fake "word" from the available characters (uniform random). Length is normal distribution with the passed parameters
	"""
	#TODO weight the characters based on sign length? (more likely to send harder chars)
	
	wordlen = max(1,int(np.round(normal(avg_wordlen, wordlen_sig),0))) # not shorter than 1 char
	return ''.join([random.choice(available_chars) for _ in range(wordlen)])


#TODO: rolling input. basically we want an operating mode where you can have it do any of these:
# - play a certain text in morse (random either from a dictionary or pure random)
# - play a random text and quiz you on it
# - super cool would be if we could implement a decoder and allow keying but that's a bit advanced for right now
# - change wpm, tone, etc settings

def play_morse_text(text,key_wpm,farn_wpm,tone_freq):
	"""
	wraps text_to_morse and play_morse to make them simpler to access
	"""
	play_morse(text_to_morse(text), key_wpm, farn_wpm, output_freq_hz = tone_freq)

class Timer:
	
	"""
	simple timer (primarily) for delays
	"""
	
	def __init__(self, timeout):
		# settings
		self.timeout = timeout
		
		# internal variables
		self.start_time = 0 # will be set when we start
		self.running = False
	
	def expired(self):
		cur_time = time()
		elapsed_time = cur_time - self.start_time
		return elapsed_time >= self.timeout
	
	def start(self):
		self.start_time = time()
		self.running = True
	
	def stop_reset(self):
		self.start_time = 0
		self.running = False
	
	def is_running(self):
		return self.running

class SoundProc:
	
	"""
	handles sound output
	"""
	
	def __init__(self):
		# settings
		self.tone_freq_hz = DEFAULT_TONE_FREQ_HZ
		self.sam_rate_hz = DEFAULT_SAM_RATE_HZ
		self.output_key_wpm = DEFAULT_KEY_WPM
		self.output_farn_wpm = DEFAULT_FARN_WPM
		
		# internal variables
		self.cw_on = False # are we currently 'on'? (so if the signal is ON, then this is true, o/w false)
		self.current_text = [] # what text (encoded) are we currently playing? (this will shrink as we get through it)
		self.current_key_unitlen = 0 # this will be set when playback starts
		self.current_farn_unitlen = 0 # also set when playback starts
		self.current_playobj = None # this will be set when we need to actually play a sound
		self.delay_timer = None # this will be set when we start a pause in playback
	
	def playing_sound(self):
		"""
		return whether a sound is currently being played (whether the speakers are going or not -- so if you're in a space this will return true)
		"""
	
		return (self.current_playobj is not None) or (self.delay_timer is not None) or (0 != len(self.current_text))
	
	def pop_soundlen_ison(self):
		"""
		pop the next character from current_text and return its soundlen/ison 
		"""
		
		if 0 == len(self.current_text):
			return None
		
		ch = self.current_text.pop(0)
		if ch not in OUTPUTS:
			raise ValueError("Character {} not in encoded outputs dict, current_text may not have been encoded: {}".format(ch, self.current_text))
		
		return OUTPUTS[ch][0], OUTPUTS[ch][1], ch
	
	def start_morse_playback(self, text):
		"""
		given an unencoded string of text, begin audio playback
		"""
	
		if self.playing_sound():
			# there is currently a sound playing (whether active or not), cancel it
			self.current_text = []
			self.cw_on = False
			
			# if we're in an 'ON', stop playback
			if self.current_playobj is not None:
				self.current_playobj.stop()
				self.current_playobj = None
				
			# if we're in a delay, reset the timer
			if self.delay_timer is not None:
				self.delay_timer = None
		
		# set up for starting a new sound
		ttm_rawout = text_to_morse(text)
		if ttm_rawout is None:
			return # don't do anything
		
		self.current_text = list(ttm_rawout)
		self.current_key_unitlen = 60 / (ASSUME_AVG_WORDLEN_UNITS * self.output_key_wpm)
		self.current_farn_unitlen = 60 / (ASSUME_AVG_WORDLEN_UNITS * self.output_farn_wpm)
		
		print("sound playing...")
		
	
	def check_playback(self):
		"""
		check for/process actively playing sounds
		"""
		
		if not self.playing_sound():
			return # nothing to do, no sound processing
		
		# three options: we haven't started at all, we're processing an OFF, or we're processing an ON
		
		if self.delay_timer is not None:
			# currently processing a pause
			if self.delay_timer.expired():
				# timer is expired, reset
				self.delay_timer = None
			else:
				# timer is not expired, nothing else to do
				return
				
		elif self.current_playobj is not None:
			# currently processing a beep
			if not self.current_playobj.is_playing():
				# then the beep has completed, reset
				self.current_playobj = None
			else:
				# the beep is still going. nothing else to do
				return
		
		# if we get here, no matter what there is no processing going on. process the next character, or return if there is none
		if 0 == len(self.current_text):
			# we're done here
			return
		
		soundlen,is_on,ch = self.pop_soundlen_ison()
		
		# figure out the length of the sound in seconds
		sound_len_sec = 0
		if ' ' == ch:
			#TODO apply this also to inter-word spacing?
			# farnsworth delay on inter-character spacing only
			sound_len_sec = soundlen * self.current_farn_unitlen
		else:
			# normal timing
			sound_len_sec = soundlen * self.current_key_unitlen
		
		# start the sound itself
		if is_on:
			# get audio data
			audio_dat = gen_sinewave(self.tone_freq_hz, sound_len_sec, sam_rate_hz=self.sam_rate_hz)
			
			# start playback
			self.current_playobj = sa.play_buffer(audio_dat, 1, 2, self.sam_rate_hz)
		
		else:
			# set up timer
			self.delay_timer = Timer(sound_len_sec)
			self.delay_timer.start()
		

class Trainer:
	
	"""
	wraps state info for the trainer object
	"""
	
	def __init__(self, sound_proc: SoundProc):
		# settings
		self.avg_rand_wordlen = DEFAULT_AVG_WORDLEN
		self.rand_wordlen_sig = DEFAULT_WORDLEN_SIG
		self.allowed_chars = list(set(MORSE_ENCS.keys()) - {' '})
		
		# ext objects
		self.sound_proc = sound_proc
		
		# internal vars
		self.current_ans = "" # answer to the current question
		self.quiz_active = False # are we waiting for the user to answer the most recent quiz?
		self.this_q_start_time = 0 # when did we ask the current question? (time of completion of sound play)
		self.num_correct = 0
		self.num_incorrect = 0
	
	def quiz_one(self):
		"""
		generate a random word, play it, and ask the user what it was
		"""
		#TODO dictionary based words instead of totally random?
		self.current_ans = gen_rand_fake_word(self.avg_rand_wordlen, self.rand_wordlen_sig, self.allowed_chars)
		self.quiz_active = True
		# wait to set start time until the audio finishes playing
		
		# start playback
		self.sound_proc.start_morse_playback(self.current_ans)
	
	def check_for_start_time(self):
		"""
		if the audio is done, start the timer
		"""
		if (0 == self.this_q_start_time) and (not self.sound_proc.playing_sound()) and (self.quiz_active):
			self.this_q_start_time = time()
			
	
	def check_ans(self, user_ans):
		"""
		check the input answer to see if it's right
		"""
		if not self.quiz_active:
			return
		
		#TODO markup misses?
		# don't care about missed spaces
		user_ans_nosp = [x for x in user_ans.upper() if x != ' ']
		actual_ans = [x for x in self.current_ans.upper() if x != ' ']
		
		if user_ans_nosp == actual_ans:
			print("Correct! ({:.2f} s)".format(time() - self.this_q_start_time))
			self.num_correct += 1
		else:
			print("Correct answer: ", self.current_ans)
			print("Your answer:    ", user_ans)
			print("Answered in {:.2f} s".format(time() - self.this_q_start_time))
			self.num_incorrect += 1
		
		print("Stats so far: {}/{} ({:.2f}%)".format(self.num_correct,
		                                             self.num_correct+self.num_incorrect,
		                                             100*(self.num_correct/(self.num_correct+self.num_incorrect))))
		
		# either way, we had an input, so reset everything
		self.current_ans = ""
		self.quiz_active = False
		self.this_q_start_time = 0
		
class Parser:
	
	"""
	parse input given by the user
	"""
	
	def __init__(self, sound_proc: SoundProc, trainer: Trainer):
		self.sound_proc = sound_proc
		self.trainer = trainer
		
		self.file_lines = [] # currently reading from a file -- these are its lines (if empty, then not currently reading)
		
		self.term_helpstr = '''
Trainger terminal. Commands:
	'<any string>' - output the string in audio (if you want a '/' at the beginning, type '//')
	'/help' - print this message
	'/q' - quit
	'/check_settings' - display values of all settings (except for allowed characters)
	'/check_allowed_chars' - list all characters that the random generator will provide, and their codes
	'/check_known_chars' - list all the characters that have known encodings
	'/qr' - play a random morse sequence according to these settings and quiz on the result
	'/set_key_wpm <number>' - set the keyer words per minute
	'/set_farn_wpm <number>' - set the keyer farnsworth wpm
	'/set_tone_freq <number>' - set the tone generator frequency (this may sound weird if the bitrate is bad)
	'/set_tone_sam_rate <number> - set the tone generator sample rate
	'/set_avg_wordlen <number>' - set the average random word length
	'/set_wordlen_sig <number>' - set the random word length stdev
	'/allow_chars <string>' - set these characters as 'allowed' for the random word generator (no delimiters)
	'/allow_prosign <string>' - set this prosign as 'allowed' for the random word generator (prosigns won't work with allow_chars) NOT IMPLEMENTED
	'/disallow_chars <string>' - set these characters as 'not allowed' for the random word generator
	'/disallow_prosign <string>' - set this prosign as 'disallowed' NOT IMPLEMENTED
	'/read_cmds <file>' - read a series of commands from a file (formatted the same as for stdin)
	'''
		
	def check_trainer_prompt(self):
		if self.trainer.quiz_active and not self.sound_proc.playing_sound():
			# then we're waiting for the user to give the answer to this one
			response = input("Enter copy: ")
			self.trainer.check_ans(response)
		
	def check_file_read(self):
		if 0 == len(self.file_lines):
			return # no file to read
		
		# call check_term_input on all lines of the file
		for line in self.file_lines:
			self.check_term_input(line)
		
		# purge file lines from memory
		self.file_lines = []
	
	def check_term_input(self, filein = None):
		def numerical_2nd(cmd):
			if len(cmd) > 1:
				cmd[1] = tryparse(cmd[1])
				if cmd[1] is None:
					return False # can't process this command
			else:
				print("ERROR, numerical second argument expected")
				return False
			return True
	
		if not self.trainer.quiz_active and not self.sound_proc.playing_sound():
			# then we should throw up the normal prompt
			response = ""
			if filein is not None: #line of the file that we're reading if not
				response = filein
			else:
				response = input(">> ")
			
			# parse this input
			# first, basic checks
			if len(response) == 0:
				return False # nothing to do
			
			if (len(response) >= 2) and ('/' == response[0]) and ('/' != response[1]):
				# then this is a command
				cmd = response[1:].split(' ')
				
				if 'check_settings' == cmd[0]:
					print( '''
	keyer: {} WPM
	keyer farnsworth: {} WPM
	tone generator freq: {} Hz
	tone generator sample rate: {} Hz
	average random word length: {}
	random word length stdev: {}
					'''.format( self.sound_proc.output_key_wpm,
					            self.sound_proc.output_farn_wpm,
					            self.sound_proc.tone_freq_hz,
					            self.sound_proc.sam_rate_hz,
					            self.trainer.avg_rand_wordlen,
					            self.trainer.rand_wordlen_sig ) )
					            
				elif 'check_allowed_chars' == cmd[0]:
					print("Current allowed chars list:")
					# sort the list first
					allowed_chars = sorted(self.trainer.allowed_chars)
					# format the list as we go
					for ch in allowed_chars:
						if ch not in MORSE_ENCS:
							print("ERROR: {} not found in morse encodings dict".format(ch))
						else:
							print("{}\t-\t{}".format(ch,MORSE_ENCS[ch.upper()]))
				
				elif 'check_known_chars' == cmd[0]:
					print("The following characters are known to the underlying encoder:")
					# sort the list first
					known_chars = sorted(list(MORSE_ENCS.keys()))
					# format as we go
					for ch in known_chars:
						if ' ' != ch:
							print("{}\t-\t{}".format(ch,MORSE_ENCS[ch]))
				
				elif 'set_key_wpm' == cmd[0]:
					if numerical_2nd(cmd):
						cmd[1] = tryparse(cmd[1])
						print("Setting keyer WPM to ", cmd[1])
						self.sound_proc.output_key_wpm = cmd[1]
				
				elif 'set_farn_wpm' == cmd[0]:
					if numerical_2nd(cmd):
						cmd[1] = tryparse(cmd[1])
						print("Setting farnsworth WPM to ", cmd[1])
						self.sound_proc.output_farn_wpm = cmd[1]
				
				elif 'set_tone_freq' == cmd[0]:
					if numerical_2nd(cmd):
						cmd[1] = tryparse(cmd[1])
						if cmd[1] > self.sound_proc.sam_rate_hz / 2:
							print("WARNING: selected frequency of {:.1f} Hz violates the nyquist sampling rate theorem limit for the current sampling rate (sampling rate: {:.0f} Hz, nyquist limit: {:.1f} Hz)".format(cmd[1],self.sound_proc.tone_freq_hz,self.sound_proc.tone_freq_hz/2))
							
						print("Setting tone frequency to {} Hz".format(cmd[1]))
						self.sound_proc.tone_freq_hz = cmd[1]
				
				elif 'set_tone_sam_rate' == cmd[0]:
					if numerical_2nd(cmd):
						cmd[1] = tryparse(cmd[1])
						# make sure simpleaudio will be okay with the selection
						if cmd[1] not in ALLOWED_AUDIO_SAM_RATES:
							print("ERROR: selected frequency of {} is not one of the non-'weird' rates. Allowed sampling rates: {}".format(cmd[1],sorted(list(ALLOWED_AUDIO_SAM_RATES))))
							
							return False
						
						print("Setting tone generator sample rate to {} Hz".format(cmd[1]))
						self.sound_proc.sam_rate_hz = int(cmd[1])
				
				elif 'set_avg_wordlen' == cmd[0]:
					if numerical_2nd(cmd):
						cmd[1] = tryparse(cmd[1])
						print("Setting average random word length to ", cmd[1])
						self.trainer.avg_rand_wordlen = cmd[1]
				
				elif 'set_wordlen_sig' == cmd[0]:
					if numerical_2nd(cmd):
						cmd[1] = tryparse(cmd[1])
						print("Setting random word length stdev to ", cmd[1])
						self.trainer.rand_wordlen_sig = cmd[1]
				
				elif 'allow_chars' == cmd[0]:
					if len(cmd) <= 1:
						print("ERROR: expected 2 arguments")
						return False # can't continue
				
					# process all chars
					for ch in cmd[1]:
						ch = ch.upper()
						if ch not in MORSE_ENCS:
							print("ERROR: {} not found in morse encodings dictionary".format(ch))
						elif ch not in self.trainer.allowed_chars and ch != ' ':
							self.trainer.allowed_chars.append(ch)
					
				elif 'disallow_chars' == cmd[0]:
					if len(cmd) <= 1:
						print("ERROR: expected 2 arguments")
						return False # can't continue
				
					# process all chars
					for ch in cmd[1]:
						ch = ch.upper()
						if ch in self.trainer.allowed_chars:
							self.trainer.allowed_chars.remove(ch)
				
				elif 'read_cmds' == cmd[0]:
					if len(cmd) <= 1:
						print("ERROR: expected 2 arguments")
						return False # can't continue
						
					# first check if the file exists
					if not isfile(cmd[1]):
						print("ERROR: file not found: ", cmd[1])
						return False # can't continue
					
					# file exists, read the lines into memory and keep going (we'll process them on the next cycle)
					with open(cmd[1],'r') as f:
						self.file_lines = [x[:-1] for x in f.readlines()] # trim \ns
				
				elif 'qr' == cmd[0]:
					# begin quiz
					self.trainer.quiz_one()
				
				elif 'q' == cmd[0]:
					return True # terminate
				
				elif 'help' == cmd[0]:
					print(self.term_helpstr)
				
				else:
					print("Unknown command: ", ''.join([x + ' ' for x in cmd]))
					print(self.term_helpstr)
				
				#no other parsing
			
			else:
				# assume the user wants this bleeped at them (but compress leading double slashes if we have them)
				if (len(response) >= 2) and ('/' == response[0]) and ('/' == response[1]):
					response = '/' + response[2:]
				
				# get encoding first
				raw_ttm_out = text_to_morse(response)
				if raw_ttm_out is None:
					return False # can't do anything
				
				self.sound_proc.start_morse_playback(response)
				# also print the word and its encoding
				print("Input:\t", response)
				print("Encoding:\t", raw_ttm_out)
		
		return False
	

class Controller:
	
	"""
	handle cycling and dispatching
	"""
	
	def __init__(self, setup_file_path):
		# objects
		self.sound_proc = SoundProc()
		self.trainer = Trainer(self.sound_proc)
		self.parser = Parser(self.sound_proc, self.trainer)
		
		# internal variables
		self.cycle_start_time = 0
		
		# read file if it was given
		if setup_file_path is not None and isfile(setup_file_path):
			# just make the parser do it when we start up
			self.parser.check_term_input("/read_cmds " + setup_file_path)
	
	def exec_one_cycle(self):
	
		self.cycle_start_time = time()
		
		# check for user input
		self.parser.check_file_read() # must be done before check_term_input
		self.parser.check_trainer_prompt() # must be done before check_term_input
		if self.parser.check_term_input():
			return True # terminate input
		
		# handle currently processing sounds
		self.sound_proc.check_playback()
		
		# check for quizzer timer starts (has to be done after check_playback)
		self.trainer.check_for_start_time()
		
		# delay the rest of the cycle
		cycle_delay(self.cycle_start_time)

# main exec logic
print("Welcome to the morse Trainger! Enter terminal input (commands are preceeded by '/') to proceed.")
init_setup_filep = None
if len(argv) > 1:
	init_setup_filep = argv[1]
else:
	print("Tip: give the path to a commands file as an argument to this script to load it automatically!")
ctrl = Controller(init_setup_filep)
while True:
	if ctrl.exec_one_cycle():
		print("terminating")
		exit(0)
