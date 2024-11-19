import cv2
import numpy as np
import inspect
import math
import distinctipy

from uuid import uuid4

from libs import params

class ParticleFilter:

    COLORS_HISTORY = [distinctipy.get_rgb256(color)[::-1] for color in distinctipy.get_colors(n_colors=25, pastel_factor=0)]
    COLORS = COLORS_HISTORY.copy()

    def __init__(self, width: int, height: int, number_of_particles: int, bounding_box: tuple) -> None:
        
        self.id = str(uuid4())

        self.width = width
        self.height = height
        self.number_of_particles = number_of_particles

        # Generate a set of particles
        self.particles = np.empty((self.number_of_particles, 2), dtype=np.float32)

        # Randomly distribute particles
        self.particles[:, 0] = np.random.uniform(0, self.width, size=self.number_of_particles)
        self.particles[:, 1] = np.random.uniform(0, self.height, size=self.number_of_particles)

        # Initialize particle weights
        self.weights = np.ones(self.number_of_particles) / self.number_of_particles

        # Calculate the centroid (assume it is the position of the object)
        x, y, w, h = bounding_box
        self.centroid = np.array([x + w / 2, y + h / 2])

        # Select one of the randomly generated colors
        if not len(ParticleFilter.COLORS):
            new_colors = [distinctipy.get_rgb256(color) for color in distinctipy.get_colors(n_colors=1, pastel_factor=0.5, exclude_colors=ParticleFilter.COLORS_HISTORY)]

            ParticleFilter.COLORS_HISTORY.extend(new_colors)
            ParticleFilter.COLORS.extend(new_colors)

        self.color = ParticleFilter.COLORS.pop()
        
        self.points = []

    def __del__(self):

        # try-except block used for handling script interruptions and the removal of the class
        try:
            # Make the color used by this particle system accessible
            ParticleFilter.COLORS.append(self.color)
        except:
            pass

    def __eq__(self, value):
        
        if isinstance(value, ParticleFilter):
            return self.id == value.id

        return False

    def __update_weights(self, centroid, measurement_noise_std=params.MEASUREMENT_NOISE_STD):

        # Calculate the distance from the measurement to each particle
        distances = np.linalg.norm(self.particles - centroid, axis=1)
        
        # Compute the likelihood of each particle (Gaussian distribution)
        self.weights[:] = np.exp(-distances**2 / (2 * measurement_noise_std**2))
        self.weights += 1.e-300  # Avoid division by zero
        self.weights /= np.sum(self.weights)  # Normalize the weights

    def __resample(self):
        
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure rounding issues don't affect the resampling
        indexes = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, self.number_of_particles))

        # Resample the particles
        self.particles[:] = self.particles[indexes]
        self.weights.fill(1.0 / self.number_of_particles)
    
    def __predict(self, stddev=params.STDDEV):

        # Add noise to the particle positions (simulates random motion)
        noise = np.random.normal(0, stddev, size=self.particles.shape)
        self.particles += noise

        # Keep particles within the frame boundaries
        self.particles[:, 0] = np.clip(self.particles[:, 0], 0, self.width)
        self.particles[:, 1] = np.clip(self.particles[:, 1], 0, self.height)
    
    def __estimate(self):

        return np.average(self.particles, weights=self.weights, axis=0)

    def step(self, centroid) -> tuple:

        # Update particle weights based on the measurement
        self.__update_weights(centroid=centroid)
        
        # Resample particles based on their weights
        self.__resample()

        # Predict the next state of the particles
        self.__predict()

        estimate_position = self.__estimate()

        return estimate_position, self.particles.copy()

    def add_point(self, point: tuple):

        if len(self.points) == params.MAX_POINTS:
            self.points.pop(0)

        self.points.append(point)
    
    def get_points(self) -> list:
        return self.points.copy()

    def get_distance(self, centroid: np.ndarray) -> int:
        return math.dist(centroid, self.centroid)

    def get_color(self):
        return tuple(list(self.color).copy())

def particle_filtering(mat: cv2.typing.MatLike | cv2.cuda.GpuMat | cv2.UMat, bounding_boxes: list[tuple], reset: bool = False) -> dict:

    height, width = mat.shape[:2]

    function = eval(inspect.stack()[0][3])

    try:
        # Check if the current function has particle systems
        function.particle_systems
    except:
        reset = True

    if reset:
        # Reset and initialize the particle systems for each bounding box
        function.particle_systems = []

        for bounding_box in bounding_boxes:
            function.particle_systems.append(ParticleFilter(width=width, height=height, number_of_particles=params.NUMBER_OF_PARTICLES, bounding_box=bounding_box))

    # Make a copy of the particle systems
    particle_systems = function.particle_systems.copy()

    # Calculate distances between each particle system and every bounding box
    particle_systems_dict = {}

    for particle_system in particle_systems:

        tmp = []

        for bounding_box in bounding_boxes:
            x, y, w, h = bounding_box
            centroid = np.array([x + w / 2, y + h / 2])
            distance = particle_system.get_distance(centroid=centroid)

            tmp.append([bounding_box, distance])
        
        tmp.sort(key=lambda x: x[1])

        particle_systems_dict[particle_system.id] = {
            "particle_system": particle_system,
            "items": tmp
        }

    # Iterate over each particle system to choose the best bounding box (if any)
    bounding_boxes_dict = {bounding_box: None for bounding_box in bounding_boxes}

    for particle_system in particle_systems_dict.values():
        particle_system, items = particle_system["particle_system"], particle_system["items"]

        for item in items:

            bounding_box, distance = item

            x, y, w, h = bounding_box
            centroid = np.array([x + w / 2, y + h / 2])
            origin = (int(centroid[0]), int(centroid[1]))

            # Discard all the distances higher than the threshold
            if distance > params.DISTANCE:
                break
            
            # Discard bounding boxes already assigned
            if bounding_boxes_dict.get(bounding_box, None) != None:
                continue

            bounding_boxes_dict[bounding_box] = particle_system
            break

    for bounding_box in bounding_boxes_dict:

        if bounding_boxes_dict.get(bounding_box, None) is not None:
            continue
        
        bounding_boxes_dict[bounding_box] = ParticleFilter(width=width, height=height, number_of_particles=params.NUMBER_OF_PARTICLES, bounding_box=bounding_box)
    
    # Update the particle systems and perform steps for each
    function.particle_systems = []
    
    results = {}

    for bounding_box in bounding_boxes_dict:
        x, y, w, h = bounding_box
        centroid = np.array([x + w / 2, y + h / 2])

        # Update the particle system
        particle_system = bounding_boxes_dict[bounding_box]
        estimate_position, particles = particle_system.step(centroid=centroid)

        # Correct the predictions
        origin = (int(centroid[0]), int(centroid[1]))
        estimated = (int(estimate_position[0]), int(estimate_position[1]))
        
        new_x = estimated[0] - origin[0]
        new_y = estimated[1] - origin[1]

        if new_x > 0:
            new_x = origin[0] - new_x
        else:
            new_x = origin[0] + new_x * -1
        
        if new_y > 0:
            new_y = origin[1] - new_y
        else:
            new_y = origin[1] + new_y * -1

        estimated = (new_x, new_y)

        # Keep track of previous positions
        particle_system.add_point(point=origin)

        # Save the particle system for future use
        function.particle_systems.append(particle_system)

        # Save results
        results[bounding_box] = {
            "origin": origin,
            "estimated": estimated,
            "particles": particles,
            "points": particle_system.get_points(),
            "color": particle_system.get_color()
        }

        # Draw bounding box
        #cv2.rectangle(frame, (x, y), (x + w, y + h), particle_system.get_color(), 2)

        # Draw the arrow associated to the estimated position
        #cv2.arrowedLine(kkk, origin, estimated, (255, 0, 255), 4, tipLength=0.25)

    return results