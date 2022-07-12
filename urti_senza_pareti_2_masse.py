import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
epsilon=float(input("inserisci il valore del coefficiente di restituzione (1=urto elastico, 0=urto completamente anelastico): "))
k=input("inserisci la potenza di 100: ")
class Square:
    def __init__(self, position, m, v):
        """Initialises disk with a position, mass and velocity and radius"""
        self.position = position
        self.m = m
        self.v = v

    def move(self):
        """Moves disk 'v' distance."""
        self.position[0] += self.v[0]
        self.position[1] += self.v[1]

    def collide(self, other):
        """Computes new velocities for two disks based on old velocities"""

        # Get other disk's properties
        other_position, other_m, other_v = other.get_properties()

        # Create a normal vector to the collision surface
        norm = np.array([other_position[0] - self.position[0], other_position[1] - self.position[1]])
        # Convert to unit vector
        unit_norm = norm / (np.sqrt(norm[0]**2 + norm[1]**2))
        # Create unit vector tagent to the collision surface
        unit_tang = np.array([-unit_norm[1], unit_norm[0]])

        # Project self disk's velocity onto unit vectors
        self_v_norm = np.dot(self.v, unit_norm)
        self_v_tang = np.dot(self.v, unit_tang)

        # Project other disk's velocity onto unit vectors
        other_v_norm = np.dot(other_v, unit_norm)
        other_v_tang = np.dot(other_v, unit_tang)

        # Use 1D collision equations to compute the disks' normal velocity (la velocità tangenziale non cambia perché le forze impulsive agiscono solo normalmente al piano di collisione)
        self_v_prime = ((self.m - epsilon*other_m) / (self.m + other_m)) * self_v_norm + (((1+epsilon) * other_m) / (self.m + other_m)) * other_v_norm

        other_v_prime = (((1+epsilon) * self.m) / (self.m + other_m)) * self_v_norm + ((other_m - epsilon*self.m) / (self.m + other_m)) * other_v_norm

        # Add the two vectors to get final velocity vectors and update.
        self.v = self_v_prime * unit_norm + self_v_tang * unit_tang
        other.set_v(other_v_prime * unit_norm + other_v_tang * unit_tang)

    def bounce_x(self, distance):
        """Bounces off and edge parallel to the y axis.
        Variable, 'distance', is distance to move back out of wall"""
        self.v[0] *= -1
        self.position[0] += distance

    def get_properties(self):
        """acquisisce posizione, massa, raggio e velocità"""
        return self.position, self.m, self.v

    def get_position(self):
        """acquisice la posizione"""
        return self.position

    def set_position(self, position):
        """imposta la nuova posizione"""
        self.position = position

    def set_v(self, v):
        """imposta la velocità dei dischi"""
        self.v = v
class Simulation:
    def __init__(self, size, save=False):
        self.size = size
        self.disks=[]
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.save = save
        # Generate list of disk objects with random values
        position1 = np.array([5, 5])
        v1 = np.array([3, 3]) 
        # Calculating mass of disk
        m1 = 1
        position2 = np.array([100, 100])
        v2 = np.array([0, 0]) 
        # Calculating mass of disk
        m2 = 1
        self.disks.append(Square(position1, m1, v1))
        self.disks.append(Square(position2, m2, v2))
    
    def update(self):
        """aggiorna la posizione e la velocità dei dischi"""
        for i, disk in enumerate(self.disks):
            disk.move()
            # Get disk's properties
            disk_position, disk_mass, disk_velocity = disk.get_properties()

            for other in self.disks[i + 1 :]:
                # Get other disk's properties
                other_position, other_mass, other_velocity = other.get_properties()
                # Find the scalar difference between the position vectors
                diff_position = np.linalg.norm(disk_position - other_position)

                # Check if the disks' radii touch. If so, collide
                if diff_position <= 10:
                    disk.collide(other)
                    diff_vector = disk_position - other_position
                    # 'clip' is how much the disks are clipped
                    clip = 10 - diff_position
                    # Creating normal vector between disks
                    diff_norm = diff_vector / diff_position
                    # Creating 'clip_vec' vector that moves disk out of the other
                    clip_vec = diff_norm * clip
                    # Set new position
                    disk.set_position(disk_position + clip_vec)
            # Check if the disk's coords is out of bounds
            # X-coord
    def center_mass(self):
        center_of_mass_position=np.zeros(2)
        center_of_mass_position=center_of_mass_position.astype('float64')
        for i, disk in enumerate(self.disks):
            # Get disk's properties
            disk_position, disk_mass, disk_velocity = disk.get_properties()
            center_of_mass_position += disk_position.astype('float64')
        return center_of_mass_position/2

    def animate(self,_):
        """aggiorna il grafico e la simulazione animandola"""
        self.ax.patches = []
        self.update()
        circle_center_of_mass=plt.Circle(self.center_mass(), 5, color="green")
        self.ax.add_patch(circle_center_of_mass)
        for disk in self.disks:
            position, disk_mass, disk_velocity = disk.get_properties()
            square = plt.Circle(position, 5, color="red")
            self.ax.add_patch(square)
        return (self.ax)

    def show(self):
        """disegna il tavolo e i dischi"""
        self.ax.set_aspect(1)
        self.ax.set_xlim(0, self.size)
        self.ax.set_ylim(0, self.size)
        animazione = animation.FuncAnimation(self.fig, self.animate, frames=150, interval=1, save_count=150, cache_frame_data=True)
        if self.save:
            writervideo = animation.PillowWriter(fps=25)
            animazione.save("animazione.gif", writer=writervideo)
        plt.show()

simulazione = Simulation(size=300, save=True)
simulazione.show()