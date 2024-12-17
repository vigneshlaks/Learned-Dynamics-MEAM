## Key Contributions and Challenges Overcome

### 1. Working with Safe Control Environment

Incorporating random controllers into the Safe Control Environment framework was extremely difficult due to the layers of abstraction in the Safe Control Gym implementation. Major challenges included:  

- Creating random initialization for LQR data generation required fundamentally understanding the codebase and its architecture.  
- Changes were necessary to enable proper data generation.  
- Navigating rigid control flow and initialization constraints required a deep understanding of the mature codebase, taking many hours of time alone.

### 2. Data Augmentation 

- Each run generated an extra observation, which had to be carefully parsed.  
- Parsing the data into observation-action-next state tuples required significant effort to ensure proper alignment and maintain data integrity. 
- Considering model architecture to balance both fidelity while maintaining computational feasibility
- Other more minor issues and bugs related to modeling and training

### 3. Incorporating Dynamics with Controller

- Coupling the Smoothed Dynamics with the controller framework provided by the Safe Control Environment proved to be complex.  
- Codebase insights gained while building the random controller were extremely helpful in overcoming these difficulties.

Arasappan, A., Lakshmanan, V., & Yu, S. (2024). *Investigating Smoothing Methods of Learned Dynamics for Trajectory Optimization in Robotic Control Tasks.* University of Pennsylvania.
