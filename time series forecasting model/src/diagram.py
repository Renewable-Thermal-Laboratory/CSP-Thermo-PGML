from graphviz import Digraph

dot = Digraph(comment='TempLSTM Architecture')

dot.attr(rankdir='LR', fontsize='12')

# Nodes
dot.node('I', 'Input\n[10×14]', shape='ellipse')
dot.node('L1', 'LSTM (256)', style='filled', fillcolor='lightgreen')
dot.node('D1', 'Dropout (0.3)', shape='box', style='dashed')
dot.node('L2', 'LSTM (128)', style='filled', fillcolor='lightgreen')
dot.node('D2', 'Dropout (0.3)', shape='box', style='dashed')
dot.node('A', 'Multihead Attention\n(128, 4 heads)', style='filled', fillcolor='mistyrose')
dot.node('FC', 'Fully Connected\n128 → 64 → Output', style='filled', fillcolor='lightblue')
dot.node('O', 'Output\n[10]', shape='ellipse')

# Edges
dot.edges([('I', 'L1'), ('L1', 'D1'), ('D1', 'L2'), ('L2', 'D2'), ('D2', 'A'), ('A', 'FC'), ('FC', 'O')])

# Save and render
dot.render('temp_lstm_architecture_new', format='png', cleanup=True)
print("✅ Diagram saved as temp_lstm_architecture_new.png")
