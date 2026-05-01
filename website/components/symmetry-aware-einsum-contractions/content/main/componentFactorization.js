const p = (text) => ({ kind: 'paragraph', text });

const componentFactorization = {
  title: 'Component Factorization',
  deck: 'When do product counts and accumulation counts factor independently?',
  slots: {
    intro: [
      p('Large einsums often decompose into independent label systems. But independence is a property of the certified group action, not merely of a picture. The safe factorization condition is $G_{\\text{pt}} \\cong \\prod_a G_a$ with $L = \\bigsqcup_a L_a$, where each factor $G_a$ acts only on $L_a$ and the global action is the direct product of those independent local actions.'),
      p('Under this certified independent direct-product decomposition, the assignment space, product-orbit count, output action, and incidence relation factor componentwise. Then $M = \\prod_a M_a$ and $\\alpha = \\prod_a \\alpha_a$. This factorization is mathematical, not merely visual. A diagonal action such as a single generator $(i\\,j)(k\\,l)$ remains coupled unless the certified group actually factors into an independent action on $\\{i,j\\}$ and an independent action on $\\{k,l\\}$.'),
    ],
    produces: [
      p('A certified decomposition into independent components: for each component $a$, the local label set $L_a$, local visible labels $V_a = V \\cap L_a$, local summed labels $W_a = W \\cap L_a$, local group $G_a$, and local inherited output action $H_a = \\mathrm{Stab}_{G_a}(V_a)|_{V_a}$. The classification tree runs on each certified independent component.'),
    ],
  },
};

export default componentFactorization;
