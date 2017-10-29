import analysis
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

# --- Run all code and save figs ---
for var, func in analysis.__dict__.items():
    if not var.startswith('plot_'):
        continue
    print(var)
    out = func()
    if not isinstance(out, plt.Figure):
        fig = out.figure
    else:
        fig = out

    fig.savefig('figures/{}.png'.format(var), bbox_inches='tight')
    
    
    