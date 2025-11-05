import numpy as np
import random
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt

# Load berlin52 dataset
def load_berlin52(path):
    data = pd.read_csv(path)
    return data[['X', 'Y']].values

def distance(a, b):
    return np.linalg.norm(a - b)

def total_distance(route, coords):
    return sum(distance(coords[route[i]], coords[route[(i + 1) % len(route)]]) for i in range(len(route)))

# Selection methods
def tournament_selection(pop, fitness, k=3):
    selected = random.sample(list(zip(pop, fitness)), k)
    return min(selected, key=lambda x: x[1])[0]

def roulette_selection(pop, fitness):
    inv_fitness = 1 / np.array(fitness)
    probs = inv_fitness / inv_fitness.sum()
    return pop[np.random.choice(len(pop), p=probs)]

# Crossover methods
def one_point_crossover(p1, p2):
    point = random.randint(1, len(p1) - 2)
    child = p1[:point] + [x for x in p2 if x not in p1[:point]]
    return child

def two_point_crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = [None]*len(p1)
    child[a:b] = p1[a:b]
    fill = [x for x in p2 if x not in child]
    j = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = fill[j]
            j += 1
    return child

# Mutation methods
def swap_mutation(route):
    route = route.copy()
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route

def bitflip_mutation(route):
    route = route.copy()
    a, b = sorted(random.sample(range(len(route)), 2))
    route[a:b] = list(reversed(route[a:b]))
    return route

# Initialize population
def init_population(size, n_cities):
    pop = [random.sample(range(n_cities), n_cities) for _ in range(size)]
    return pop

# Map string choices to functions
SELECTION_METHODS = {
    "Tournament": tournament_selection,
    "Roulette": roulette_selection
}

CROSSOVER_METHODS = {
    "One-Point": one_point_crossover,
    "Two-Point": two_point_crossover
}

MUTATION_METHODS = {
    "Swap": swap_mutation,
    "Bit-Flip": bitflip_mutation
}

def plot_route(coords, route):
    """Plot the TSP route"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot cities
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    
    # Plot route
    route_coords = coords[route + [route[0]]]
    ax.plot(route_coords[:, 0], route_coords[:, 1], 'b-', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Best Route Found')
    ax.grid(True, alpha=0.3)
    
    return fig

# Global variable to store testing results
testing_results = []

def load_testing_results():
    """Load existing testing results from CSV"""
    try:
        df = pd.read_csv("results.csv")
        if df.empty:
            return []
        return df.to_dict('records')
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []

def save_testing_results(results):
    """Save testing results to CSV"""
    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)

# Load existing results on startup
testing_results = load_testing_results()

def run_genetic_algorithm(pop_size, generations, mutation_rate, 
                          selection_method, crossover_method, mutation_method,
                          progress=gr.Progress()):
    """Run the genetic algorithm with specified parameters"""
    
    # Load data from repository
    coords = load_berlin52("berlin52.csv")
    
    # Get method functions
    selection_fn = SELECTION_METHODS[selection_method]
    crossover_fn = CROSSOVER_METHODS[crossover_method]
    mutation_fn = MUTATION_METHODS[mutation_method]
    
    # Initialize
    pop = init_population(pop_size, len(coords))
    history = []
    
    # Evolution
    for gen in progress.tqdm(range(generations), desc="Evolving"):
        fitness = [total_distance(p, coords) for p in pop]
        new_pop = []
        
        for _ in range(pop_size):
            parent1 = selection_fn(pop, fitness)
            parent2 = selection_fn(pop, fitness)
            child = crossover_fn(parent1, parent2)
            
            if random.random() < mutation_rate:
                child = mutation_fn(child)
            
            new_pop.append(child)
        
        pop = new_pop
        best = min(fitness)
        history.append(best)
    
    # Final results
    final_fitness = [total_distance(p, coords) for p in pop]
    best_idx = np.argmin(final_fitness)
    best_route = pop[best_idx]
    best_distance = final_fitness[best_idx]
    
    # Create history dataframe
    history_df = pd.DataFrame({
        'Generation': range(1, generations + 1),
        'Best Distance': [f"{d:.2f}" for d in history]
    })
    
    # Plot convergence
    fig_conv, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, linewidth=2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Distance')
    ax.set_title('Convergence Over Generations')
    ax.grid(True, alpha=0.3)
    
    # Plot route
    fig_route = plot_route(coords, best_route)
    
    # Results text
    results = f"**Best Distance: {best_distance:.2f}**"
    
    # Add to testing results
    testing_results.append({
        'Population Size': pop_size,
        'Generations': generations,
        'Mutation Rate': mutation_rate,
        'Selection': selection_method,
        'Crossover': crossover_method,
        'Mutation': mutation_method,
        'Best Distance': f"{best_distance:.2f}"
    })
    
    # Save to CSV file
    save_testing_results(testing_results)
    
    testing_df = pd.DataFrame(testing_results)
    
    return fig_conv, fig_route, results, history_df, testing_df

# Create Gradio interface
with gr.Blocks(title="GA TSP Solver") as demo:
    gr.Markdown("# Genetic Algorithm - TSP Solver")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Parameters")
            pop_size = gr.Slider(10, 200, value=50, step=10, label="Population Size")
            generations = gr.Slider(50, 1000, value=300, step=50, label="Generations")
            mutation_rate = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Mutation Rate")
            
            selection = gr.Radio(["Tournament", "Roulette"], value="Tournament", label="Selection")
            crossover = gr.Radio(["One-Point", "Two-Point"], value="Two-Point", label="Crossover")
            mutation = gr.Radio(["Swap", "Bit-Flip"], value="Bit-Flip", label="Mutation")
            
            run_btn = gr.Button("Run", variant="primary")
        
        with gr.Column(scale=2):
            results_text = gr.Markdown()
            with gr.Row():
                convergence_plot = gr.Plot(label="Convergence")
                route_plot = gr.Plot(label="Best Route")
    
    with gr.Accordion("Generation History", open=False):
        history_table = gr.Dataframe(headers=["Generation", "Best Distance"], interactive=False)
    
    gr.Markdown("### Testing")
    
    # Load initial testing data
    initial_testing_data = pd.DataFrame(testing_results) if testing_results else pd.DataFrame(
        columns=["Population Size", "Generations", "Mutation Rate", "Selection", "Crossover", "Mutation", "Best Distance"]
    )
    
    testing_table = gr.Dataframe(
        headers=["Population Size", "Generations", "Mutation Rate", "Selection", "Crossover", "Mutation", "Best Distance"],
        interactive=False,
        value=initial_testing_data
    )
    
    run_btn.click(
        fn=run_genetic_algorithm,
        inputs=[pop_size, generations, mutation_rate, selection, crossover, mutation],
        outputs=[convergence_plot, route_plot, results_text, history_table, testing_table]
    )

if __name__ == "__main__":
    demo.launch()