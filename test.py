from graphviz import Graph

# Create an undirected graph
dot = Graph()

# Add variable node (e.g., X0 for the unknown pose)
dot.node("X0", "Variable: X0", shape="ellipse", color="lightblue", style="filled")

# Add factor nodes with names
dot.node("CustomFactor", "Custom Factor: Relative Pose", shape="box", color="lightgreen", style="filled")
dot.node("PriorFactor", "Prior Factor", shape="box", color="lightyellow", style="filled")

# Connect the factors to the variable
dot.edge("CustomFactor", "X0", label="Constraint 1")
dot.edge("PriorFactor", "X0", label="Constraint 2")

# Render the graph
dot.render("factor_graph_with_labels", format="png", cleanup=True)