
# RA, 2019-07-20

import pandas as pd
import matplotlib.pyplot as plt



PARAM = {
	'input_graph': "../dbeaver/eventcount_201907202337.csv",

	'fig': "OUTPUT/eventcount/eventcount.{ext}",
}



df = pd.read_csv(PARAM['input_graph'])

(fig, ax) = plt.subplots()
fig: plt.Figure
ax: plt.Axes

df0 = df.loc[df['ab_slot1_variant'] == 'Control', :]
ax.loglog(df0['event_count'], df0['freq'], label="Partner A -- Control")

df0 = df.loc[df['ab_slot1_variant'] == 'Test', :]
ax.loglog(df0['event_count'], df0['freq'], label="Partner A -- Test")

df0 = df.loc[df['partner_key'] == 'Partner B', :]
ax.loglog(df0['event_count'], df0['freq'], label="Partner B")

ax.legend()

ax.set_xlabel("Events per distinct user")
ax.set_ylabel("Distinct users")

for ext in ['png', 'eps']:
	fig.savefig(
		PARAM['fig'].format(ext=ext),
		bbox_inches='tight', pad_inches=0,
		dpi=300
	)

# plt.show()
plt.close(fig)
