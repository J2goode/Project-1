For our first project, we were tasked with finding the optimal path for a hypothetical rocket to land on a platform. We could be as detailed with our process and
modeling as we wanted, as well as use any method we wanted. For this reason, I chose to focus on making the rocket problem as close to actual conditions as possible, 
while moving in one dimension (in this case, the y direction). I also wanted to make this solution as robust as possible, so I also attempted to minimize the loss 
function for as large of a range of initial states as possible.
To make the problem more representative of real life conditions, I created a free body diagram to account for all forces acting on the system in the vertical direction.
This showed that the force due to drag acted in tandem with the boosters to counteract the force of gravity. These gave the downward acceleration of the system, which 
the rest of the sample code was able to handle with no changes. To account for the height of the landing platform, I had to subtract the platform height from the current 
height of the system. To account for a range of initial states, I had to adjust the initialize state function in the sample code. Instead of an initial state of [0,0], I 
made the initial position a random height between the platform and 11m.

# Analysis
From tweaking constants, I found that the boost acceleration had the largest effect on whether the loss function went to zero or it became undefined. In most 
instances, when the boost acceleration was less than the gravity acceleration, the optimization could not be completed. The opposite was also true; if the boost 
acceleration was too large, it would outpace gravity easily and there would be infinite solutions. While you could change values such as the drag coefficient or the air 
density, in the real world they are not as easily changed as the maximum boost acceleration and weight of the system are.

The function would not converge in other cases as well. From multiple passes, the system struggled to find optimal solutions with certain floats, even when it
was fully capable of solving for much larger initial states. Occasionally, the optimization would yield no change in the loss function, staying high throughout. This 
also occurred frequently when an initial velocity was applied; under the current time constraint I was unable to devise a solution to this, so initial velocities were 
ignored for this project.

# Conclusion
As a whole, a gradient descent based neural network was easy to implement, relatively robust in delivering solutions, and quick acting. Training our model over
40 cycles took less than 1 minute to complete and consistently delivered the expected results. The code provided above was able to solve for the optimal solutions within
the space for a range of initial states and platform heights. This code is able to be used for many more applications than what was shown in this project. With minimal
tweaks to the dynamics to account for friction, this could be used to find the appropriate braking force to apply for automatic brake systems in modern cars.
