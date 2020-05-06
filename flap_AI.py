'''
	FLAP_AI:
		Built using neat-python

	NEURAL NETWORK INFO:
		INPUT - Bird.y, TOP_PIPE, BOTTOM_PIPE
		OUTPUT - Jump?
		ACTIVATION FUNTION - tanh
		POPULATION - 100 birds
		FITNESS FUNCTION:
			a method to evaluate the performance of the bird
			--> Distance covered (say)

	TRANSLATING CONFIG FILE:

		(refer neat-python config)
		MAX GENERATION - 30
		fitness_criterion     = max -- CHOOSING THE BIRD WITH MAX FITNESS
		fitness_threshold     = 100 -- Score to achieve
		pop_size              = 50 -- population
		reset_on_extinction   = False -- when all the birds die at a time then restart (not req.)

		[DefaultGenome]
		# node activation options
		activation_default      = tanh -- activation func
		activation_mutate_rate  = 0.0  -- no mutation req.
		activation_options      = tanh

		# network parameters
		num_hidden              = 0 -- no hidden neuron initially we could add one 
		num_inputs              = 3 -- ref. INPUT
		num_outputs             = 1 -- ref OUTPUT
'''

import pygame
import neat
import time
import os
import random
import pickle
pygame.font.init()

WIN_WIDTH = 600
WIN_HEIGHT = 800

STAT_FONT = pygame.font.SysFont("comicsans", 50)

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

gen = 0

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png"))) for x in range(1,4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())
# transform.scale2x doubles image size

class Bird:
	IMGS = bird_images
	MAX_ROTATION = 25
	ROT_VEL = 20
	ANIMATION_TIME = 5

	'''
	rotation of the beak of the bird upon each jump
	velocity of the rotation itself
	animation time per frame
	'''

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.tilt = 0
		self.tick_count = 0
		self.vel = 0
		self.height = self.y
		self.img_count = 0
		self.img = self.IMGS[0]

	def jump(self):
		self.vel = -10.5
		self.tick_count = 0
		self.height = self.y

		'''
		The pygame window starts with a (0,0) in the top left corner
		Therefore for a upward movment we must travel in the negative axis
		Hence the negative vslue fot jump

		tick_count keeps track of the last jump
		'''

	def move(self):
		self.tick_count += 1
		d = self.vel*self.tick_count + 0.5*(3)*self.tick_count**2

		'''
		Equation of a projectile motion in a parabola
		for one jump(tick_count = 1)
		-10.5 + 1.5 = -9

		9 pixels in the upward direction

		'''

		if d >= 16:
			d = (d/abs(d)) * 16

		if d < 0:
			d -= 2

		'''
		If the bisd is going below 16 it doesnt make sense so restrict the movement to 16
		Similarly while moing upwards as long as it doesnt hit the ceiling increase the displacement
		so as to smoothen the jump.
		'''

		self.y = self.y + d

		if d < 0 or self.y < self.height + 50:
			'''
			As along as the bird has jumped and is above the initial mark
			do not rotate the bird in the downward direction

			once it goes below rotate
			'''
			if self.tilt < self.MAX_ROTATION:
				self.tilt = self.MAX_ROTATION
		else:
			if self.tilt  > -90:
				self.tilt -= self.ROT_VEL


	def draw(self, win):
		self.img_count += 1

		if self.img_count <= self.ANIMATION_TIME:
			self.img = self.IMGS[0]
		elif self.img_count <= self.ANIMATION_TIME*2:
			self.img = self.IMGS[1]
		elif self.img_count <= self.ANIMATION_TIME*3:
			self.img = self.IMGS[2]
		elif self.img_count <= self.ANIMATION_TIME*4:
			self.img = self.IMGS[1]
		elif self.img_count == self.ANIMATION_TIME*4 +1:
			self.img = self.IMGS[0]
			self.img_count = 0

		'''
		When bird is tilting downwards then the bird need not flap it wings
		KEPT THE LEVEL IMAGE (flat wings)
		'''
		if self.tilt <= -80:
			self.img = self.IMGS[1]
			self.img_count = self.ANIMATION_TIME*2

		#rotation of image around is taken from stack overflow
		rotated_image = pygame.transform.rotate(self.img, self.tilt)
		new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
		win.blit(rotated_image, new_rect.topleft)

	def get_mask(self):
		'''
		refer Pipe.collide()
		'''
		return pygame.mask.from_surface(self.img)

class Pipe:
	GAP = 200
	VEL = 5

	def __init__(self, x):
		self.x = x
		self.height = 0
		self.gap = 200

		self.top = 0
		self.bottom = 0
		self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
		self.PIPE_BOTTOM = pipe_img

		self.passed = False
		self.set_height()

	def set_height(self):
		self.height = random.randrange(50, 450)
		self.top = self.height - self.PIPE_TOP.get_height()
		self.bottom = self.height + self.gap

	def move(self):
		self.x -= self.VEL

	def draw(self, win):
		win.blit(self.PIPE_TOP, (self.x, self.top))
		win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

	def collide(self, bird):
		'''
		MASKING:
			For collision of two objects an imaaginary box around each object 
			is drawn but then the real time collision will not be visible
			hence we use masking to make sure the objects actually collide and
			not the surrounding.

		OFFSET:
			distance between the masks

		CALCULATE POINT OF COLLISION (POC):
			mask.overlap() --> returns the value of the poc, NONE if it does not collide
			if return value not null(None) then collision has happened
		'''
		
		bird_mask = bird.get_mask()
		top_mask = pygame.mask.from_surface(self.PIPE_TOP)
		bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

		top_offset = (self.x - bird.x, self.top - round(bird.y))
		bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))


		b_point = bird_mask.overlap(bottom_mask, bottom_offset)
		t_point = bird_mask.overlap(top_mask, top_offset)

		if t_point or b_point:
			return True

		return False

class Base:
	'''
	Draw two images for a frame  x1 , x2
	both images move at the same velocity towards the left
	Once the image hits the end then it cycles back to the first position.

		----|----
	   ----|----
	  ----|----
	 ----|---- 
		:
		.
		:
		----|----|
	'''
	VEL = 5 #same as pipe
	WIDTH = base_img.get_width()
	IMG = base_img

	def __init__(self, y):
		self.y = y
		self.x1 = 0
		self.x2 = self.WIDTH

	def move(self):
		self.x1 -= self.VEL
		self.x2 -= self.VEL 

		if self.x1 + self.WIDTH < 0:
			self.x1 = self.x2 + self.WIDTH

		if self.x2 + self.WIDTH < 0:
			self.x2 = self.x1 + self.WIDTH

	def draw(self, win):
		win.blit(self.IMG, (self.x1, self.y))
		win.blit(self.IMG, (self.x2, self.y))

def draw_window(win, birds, pipes, base, score, gen):

	if gen == 0:
		gen = 1

	win.blit(bg_img, (0, 0))

	for pipe in pipes:
		pipe.draw(win)

	text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
	win.blit(text, (WIN_WIDTH - 1- text.get_width(), 10))
    
	text = STAT_FONT.render("GEN: " + str(gen), 1, (255, 255, 255))
	win.blit(text, (10, 10))
	

	base.draw(win)

	for bird in birds:
		bird.draw(win)

	pygame.display.update()

def eval_genomes(genomes, config):
	global gen

	gen += 1

	nets = [] # each genome(ge) are essentially neural nets
	ge = [] #to keep track of the returning birds
	birds = []
	'''
	genome is a tuple with (genome_id, <genome itself>)
	'''
	for _, g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		birds.append(Bird(230, 350))
		g.fitness = 0
		ge.append(g)

	bird = Bird(230, 350)
	base = Base(730)
	pipes = [Pipe(700)]

	win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	score = 0

	clock = pygame.time.Clock()

	run = True
	while run:
		clock.tick(30)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()

		pipe_ind = 0
		if len(birds) > 0:
			if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
				pipe_ind = 1
		else:
			'''
			Stop gen when no birds alive
			'''
			run = False
			break
		'''
		To fix a runtime bug: to index the pipes
		if the bird has passed the pipe the index is 1 (second pipe)
		else index 0
		''' 

		for x,bird in enumerate(birds):
			bird.move()
			ge[x].fitness += 0.1
			# this is to give points for the bird for staying alive (generated every second)

			output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
			#output is a list of neurons but ours is only of one ref. OUTPUT
			if output[0] > 0.5:
				bird.jump()

		#bird.move()
		add_pipe =False
		base.move()

		rem = [] # list to store the pipes out of the frame
		for pipe in pipes:
			for x, bird in enumerate(birds):
				if pipe.collide(bird): 
					'''
					remove birds that collide
						reduce the fitness paramter of the bird
					'''
					ge[x].fitness -= 1
					birds.pop(x)
					nets.pop(x)
					ge.pop(x)

				if not pipe.passed and pipe.x < bird.x:
					pipe.passed = True
					add_pipe = True

			if pipe.x + pipe.PIPE_TOP.get_width() < 0:
				rem.append(pipe)

			pipe.move()

		if add_pipe:
			score += 1
			for g in ge:
				'''
				Note that we are not increasing the fitness by one
				so as to ensure that the bird is forced to go through the pipes(in between)
				rather than just being motivated to move to the next stage(generation).
				'''
				g.fitness += 5
			pipes.append(Pipe(700))

		for r in rem:
			pipes.remove(r)

		for x,bird in enumerate(birds):
			if bird.y + bird.img.get_height() >= 730 or bird.y < 0: 
				'''
				first constraint for hitting floor
				or constraint for hitting the top
				which is not elimination in the game but the network tend to shoot up the bird
				so that it never hit the pipe
				'''
				birds.pop(x)
				nets.pop(x)
				ge.pop(x)

		draw_window(win, birds, pipes, base, score, gen)

def run(config_file):
	'''
	NEAT recomendation ref. documentation 
	'''

	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
	                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
	                     config_file)

	p = neat.Population(config)

	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	winner = p.run(eval_genomes, 50) #no of genrations to be run = 50
	'''
	To be sent the population values to the "main" which is the fitness function
	for 50 times 
	'''
	print('\nBest genome:\n{!s}'.format(winner))


if __name__ == "__main__":
	#ref run(config_path):
	local_dir  = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-feedforward.txt")
	run(config_path)
