from graphviz import Source, Graph
from gtsam import NonlinearFactorGraph, Values


class FactorGraphVisualization:


    @staticmethod
    def format_to_graphviz(graph: NonlinearFactorGraph, values: Values) -> Graph:
        dot = Graph()

        value_symbol = 'x'
        for key in values.keys():
            dot.node(value_symbol + str(key), shape='ellipse')

        for i in range(graph.size()):
            factor = graph.at(i)
            factor_name = f'Factor {i}'
            dot.node(factor_name, type(factor).__name__, shape='box', style='filled')
            for key in factor.keys():
                dot.edge(factor_name, value_symbol + str(key))
        return dot


    @classmethod
    def draw_factor_graph(clc, output_folder: str, graph: NonlinearFactorGraph, values: Values, filename: str = 'factor_graph') -> None:
        s = clc.format_to_graphviz(graph, values)
        s.render(filename, format='png', cleanup=True, directory=output_folder)