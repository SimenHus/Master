from graphviz import Source, Graph
from gtsam import NonlinearFactorGraph, Values, Symbol


class FactorGraphVisualization:

    @staticmethod
    def format_to_graphviz(graph: NonlinearFactorGraph, values: Values) -> Graph:
        dot = Graph(engine='sfdp')

        for key in values.keys():
            sym = Symbol(key)
            name, index = chr(sym.chr()), str(sym.index())
            dot.node(name + index, shape='ellipse')

        for i in range(graph.size()):
            factor = graph.at(i)
            factor_name = f'Factor {i}'
            dot.node(factor_name, type(factor).__name__, shape='box', style='filled')
            for key in factor.keys():
                sym = Symbol(key)
                name, index = chr(sym.chr()), str(sym.index())
                dot.edge(factor_name, name + index)
        return dot


    @classmethod
    def draw_factor_graph(clc, output_folder: str, graph: NonlinearFactorGraph, values: Values, filename: str = 'factor_graph') -> None:
        s = clc.format_to_graphviz(graph, values)
        s.render(filename, format='png', cleanup=True, directory=output_folder)