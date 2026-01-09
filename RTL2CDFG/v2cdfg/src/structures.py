class Subgraph:
    def __init__(self, graph, name) -> None:
        self.G = graph
        self.name = name  # Subgraph name.
        self.successors = []  # Successor subgraphs.
        self.predecessors = []  # Predecessor subgraphs.

    def add_successor(self, successor_subgraph):
        """Add a successor subgraph."""
        if successor_subgraph not in self.successors:
            self.successors.append(successor_subgraph)
            successor_subgraph.add_predecessor(self)  # Keep predecessor list in sync.

    def add_predecessor(self, predecessor_subgraph):
        """Add a predecessor subgraph."""
        if predecessor_subgraph not in self.predecessors:
            self.predecessors.append(predecessor_subgraph)

    def __repr__(self):
        return f"Subgraph(name={self.name}, successors={len(self.successors)}, predecessors={len(self.predecessors)})"
    
    def __hash__(self):
        """Generate a unique hash from the subgraph name."""
        return hash(self.name)

    def __eq__(self, other):
        """Compare two subgraphs for equality."""
        if isinstance(other, Subgraph):
            return self.name == other.name
        return False
