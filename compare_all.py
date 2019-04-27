

while True:
    ind = np.random.choice(X_test.shape[0])
    xorg, y0 = X_test[ind], y_test[ind]

    xorg = np.expand_dims(xorg, axis=0)
    z0 = np.argmax(y0)
    z1 = np.argmax(predict(sess, env, xorg))

    if z0 != z1:
        continue

    xadvs = [make_fgsm(sess, env, xorg, eps=0.02, epochs=10),
             make_jsma(sess, env, xorg, eps=0.5, epochs=40),
             make_deepfool(sess, env, xorg, epochs=1)]
    y2 = [predict(sess, env, xi).flatten() for xi in xadvs]
    p2 = [np.max(yi) for yi in y2]
    z2 = [np.argmax(yi) for yi in y2]

    if np.all([z0 != z2]):
        break

fig = plt.figure(figsize=(4.2, 2.2))
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.1], wspace=0.01,
                       hspace=0.01)
label = ['Clean', 'FGM', 'JSMA', 'DeepFool']

xorg = np.squeeze(xorg)
xadvs = [xorg] + xadvs
xadvs = [np.squeeze(e) for e in xadvs]

p2 = [np.max(y0)] + p2
z2 = [z0] + z2

for i in range(len(label)):
    x = xadvs[i]

    ax = fig.add_subplot(gs[0, i])
    ax.imshow(x, cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel(label[i])
    ax.xaxis.set_label_position('top')

    ax = fig.add_subplot(gs[1, i])
    img = ax.imshow(x-xorg, cmap='RdBu_r', vmin=-1, vmax=1,
                    interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel('{0} ({1:.2f})'.format(z2[i], p2[i]), fontsize=12)

ax = fig.add_subplot(gs[1, 4])
dummy = plt.cm.ScalarMappable(cmap='RdBu_r',
                              norm=plt.Normalize(vmin=-1, vmax=1))
dummy.set_array([])
fig.colorbar(mappable=dummy, cax=ax, ticks=[-1, 0, 1], ticklocation='right')

print('\nSaving figure')

gs.tight_layout(fig)
os.makedirs('../out', exist_ok=True)
plt.savefig('../out/compare.png')
