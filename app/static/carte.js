/*
 * Carte choroplethe autonome (adapte du mecanisme de dashboard-complete V.html, sans son etat
 * global) -- charge un GeoJSON de departements + un JSON de valeurs par dept_code, peint.
 * Carte decorative/complementaire : le tableau accessible reste la version de reference (RGAA),
 * genere cote serveur dans le meme gabarit.
 *
 * Cache par URL : le GeoJSON departements (9,5 Mo) est partage par plusieurs cartes sur une
 * meme page (ex. eau/electricite/fibre) -- sans ce cache, chacune le re-telechargeait et le
 * re-parsait independamment (28+ Mo de JSON.parse synchrone), ce qui gelait le rendu du
 * navigateur. Un seul fetch par URL, reutilise par toutes les cartes qui le referencent.
 */
const _cacheGeojson = {};
function _chargerGeojson(url) {
    if (!_cacheGeojson[url]) {
        _cacheGeojson[url] = d3.json(url);
    }
    return _cacheGeojson[url];
}

function dessinerCarteChoroplethe(conteneurId, geojsonUrl, donnees, options) {
    const conteneur = document.getElementById(conteneurId);
    if (!conteneur) return;
    const largeur = conteneur.clientWidth || 700;
    const hauteur = Math.round(largeur * 0.72);

    const mode = options.mode || 'sequentiel'; // 'sequentiel' (taux) ou 'divergent' (erreur)
    const couleurBase = options.couleurBase || '#2a78d6';
    const titre = options.titre || '';

    d3.select(conteneur).selectAll('*').remove();

    if (options.modele) {
        d3.select(conteneur).append('p')
            .attr('class', 'carte-modele')
            .html(`<strong>Modèle :</strong> ${options.modele}`);
    }

    const svg = d3.select(conteneur).append('svg')
        .attr('viewBox', `0 0 ${largeur} ${hauteur}`)
        .attr('role', 'img')
        .attr('aria-label', titre + ' — voir le tableau ci-dessous pour les valeurs exactes');

    const groupe = svg.append('g');
    const projection = d3.geoMercator();
    const chemin = d3.geoPath(projection);

    _chargerGeojson(geojsonUrl).then(geo => {
        projection.fitSize([largeur, hauteur], geo);

        let echelleCouleur, maxAbs;
        if (mode === 'sequentiel') {
            const [dMin, dMax] = options.domaineSequentiel || [0, 100];
            echelleCouleur = d3.scaleSequential(d3.interpolate('#eef2f5', couleurBase)).domain([dMin, dMax]);
        } else {
            maxAbs = d3.max(Object.values(donnees), d => Math.abs(d.valeur)) || 20;
            // domaine ascendant [-maxAbs, 0, +maxAbs] : interpolateRdBu(0)=rouge sur la borne
            // negative (sous-estimation), interpolateRdBu(1)=bleu sur la borne positive
            // (sur-estimation) -- coherent avec le sens de "erreur = predit - reel"
            echelleCouleur = d3.scaleDiverging(d3.interpolateRdBu).domain([-maxAbs, 0, maxAbs]);
        }

        const tooltip = d3.select('body').append('div')
            .attr('class', 'carte-tooltip')
            .style('opacity', 0);

        groupe.selectAll('path')
            .data(geo.features)
            .join('path')
            .attr('d', chemin)
            .attr('fill', d => {
                const code = d.properties.PolyCode;
                const v = donnees[code];
                if (!v || v.n < (options.seuilRestitution || 0)) return '#e5e7eb';
                return echelleCouleur(v.valeur);
            })
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 1)
            .on('mouseover focus', function (event, d) {
                const code = d.properties.PolyCode;
                const v = donnees[code];
                const nom = (v && v.nom) || d.properties.PolyLabel || code;
                let contenu;
                if (!v) {
                    contenu = `<strong>${nom}</strong><br>Hors périmètre (zone d'apprentissage ou non couverte)`;
                } else if (v.n < (options.seuilRestitution || 0)) {
                    contenu = `<strong>${nom}</strong><br>Non restituable (effectif insuffisant)`;
                } else {
                    contenu = `<strong>${nom}</strong><br>${options.etiquetteValeur(v)}`;
                }
                tooltip.html(contenu).style('opacity', 1)
                    .style('left', (event.pageX + 12) + 'px')
                    .style('top', (event.pageY - 12) + 'px');
                d3.select(this).attr('stroke', '#0b0b0b').attr('stroke-width', 1.5);
            })
            .on('mouseout blur', function () {
                tooltip.style('opacity', 0);
                d3.select(this).attr('stroke', '#ffffff').attr('stroke-width', 1);
            })
            .append('title')
            .text(d => {
                const code = d.properties.PolyCode;
                const v = donnees[code];
                const nom = (v && v.nom) || d.properties.PolyLabel || code;
                return v ? `${nom} : ${options.etiquetteValeur(v)}` : nom;
            });

        // ---- legende (degrade de l'echelle reellement utilisee + etiquettes min/max) ----
        const legende = d3.select(conteneur).append('div').attr('class', 'legende-carte');
        if (mode === 'sequentiel') {
            const [dMin, dMax] = options.domaineSequentiel || [0, 100];
            const fmt = options.formatLegende || (v => v + ' %');
            legende.append('span').attr('class', 'degrade-legende')
                .style('background', `linear-gradient(to right, #eef2f5, ${couleurBase})`);
            legende.append('span').text(fmt(dMin));
            legende.append('span').text('→');
            legende.append('span').text(fmt(dMax));
        } else {
            legende.append('span').attr('class', 'degrade-legende')
                .style('background', `linear-gradient(to right, ${d3.interpolateRdBu(0)}, #f5f5f5, ${d3.interpolateRdBu(1)})`);
            legende.append('span').text(`sous-estimé, -${maxAbs.toFixed(0)} pts →`);
            legende.append('span').text(`sur-estimé, +${maxAbs.toFixed(0)} pts`);
        }
        legende.append('span').style('margin-left', 'auto').style('color', 'var(--texte-attenue)')
            .text('gris = hors périmètre (zone d\'apprentissage) ou non restituable');
    }).catch(err => {
        conteneur.innerHTML = '<p>Carte indisponible — voir le tableau ci-dessous.</p>';
        console.error('Erreur chargement carte:', err);
    });
}

/*
 * Diagramme en barres groupees (provinces x reseaux) avec mise en avant au survol : survoler la
 * legende ou une barre d'un reseau fait ressortir ce reseau sur toutes les provinces et grise les
 * deux autres (pattern "emphasis"). Tableau accessible = equivalent de reference (RGAA).
 */
const RESEAUX_GB = [
    { cle: 'eau', label: 'Eau', couleur: '#2a78d6' },
    { cle: 'electricite', label: 'Électricité', couleur: '#eb6834' },
    { cle: 'fibre', label: 'Fibre', couleur: '#1baf7a' },
];

function dessinerGraphiqueBarresGroupees(conteneurId, donnees, options) {
    options = options || {};
    const conteneur = document.getElementById(conteneurId);
    if (!conteneur) return;
    const marge = { haut: 20, droite: 20, bas: 90, gauche: 44 };
    const largeurTotale = conteneur.clientWidth || 800;
    const hauteurTotale = 420;
    const largeur = largeurTotale - marge.gauche - marge.droite;
    const hauteur = hauteurTotale - marge.haut - marge.bas;

    d3.select(conteneur).selectAll('*').remove();

    // legende, au-dessus du graphique -- toggle-to-emphasize
    const legende = d3.select(conteneur).append('div').attr('class', 'legende-barres-groupees');
    const svg = d3.select(conteneur).append('svg')
        .attr('viewBox', `0 0 ${largeurTotale} ${hauteurTotale}`)
        .attr('role', 'img')
        .attr('aria-label', 'Taux estimé par province et par réseau — voir le tableau ci-dessous pour les valeurs exactes');
    const g = svg.append('g').attr('transform', `translate(${marge.gauche},${marge.haut})`);

    const provinces = donnees.map(d => d.province);
    const x0 = d3.scaleBand().domain(provinces).range([0, largeur]).paddingInner(0.3).paddingOuter(0.1);
    const x1 = d3.scaleBand().domain(RESEAUX_GB.map(r => r.cle)).range([0, x0.bandwidth()]).paddingInner(0.12);
    const y = d3.scaleLinear().domain([0, 100]).range([hauteur, 0]);

    // grille horizontale (recessive) + axe Y
    g.append('g').attr('class', 'grille-y')
        .call(d3.axisLeft(y).ticks(5).tickSize(-largeur).tickFormat(''))
        .call(sel => sel.select('.domain').remove())
        .call(sel => sel.selectAll('line').attr('stroke', 'var(--grille)'));
    g.append('g').call(d3.axisLeft(y).ticks(5).tickFormat(d => d + ' %'))
        .call(sel => sel.select('.domain').attr('stroke', 'var(--couleur-bordure)'))
        .selectAll('text').attr('fill', 'var(--texte-secondaire)').style('font-size', '0.78rem');

    // axe X (provinces), labels inclines pour tenir dans l'espace
    const axeX = g.append('g').attr('transform', `translate(0,${hauteur})`)
        .call(d3.axisBottom(x0));
    axeX.select('.domain').attr('stroke', 'var(--couleur-bordure)');
    axeX.selectAll('text')
        .attr('fill', 'var(--texte-secondaire)').style('font-size', '0.78rem')
        .attr('transform', 'rotate(-30)').attr('text-anchor', 'end').attr('dx', '-0.4em').attr('dy', '0.3em');

    const tooltip = d3.select('body').append('div').attr('class', 'carte-tooltip').style('opacity', 0);

    function appliquerEmphase(cleActive) {
        g.selectAll('rect.barre-gb')
            .transition().duration(120)
            .attr('opacity', d => (!cleActive || d.reseau === cleActive) ? 1 : 0.2);
        legende.selectAll('.item-legende-gb')
            .transition().duration(120)
            .style('opacity', d => (!cleActive || d.cle === cleActive) ? 1 : 0.4);
    }

    legende.selectAll('.item-legende-gb')
        .data(RESEAUX_GB)
        .join('span')
        .attr('class', 'item-legende-gb')
        .attr('tabindex', '0')
        .attr('role', 'button')
        .html(d => `<span class="puce" style="background:${d.couleur}"></span>${d.label}`)
        .on('mouseenter focus', (event, d) => appliquerEmphase(d.cle))
        .on('mouseleave blur', () => appliquerEmphase(null));

    const groupes = g.selectAll('.groupe-province')
        .data(donnees)
        .join('g')
        .attr('class', 'groupe-province')
        .attr('transform', d => `translate(${x0(d.province)},0)`);

    groupes.selectAll('rect.barre-gb')
        .data(d => RESEAUX_GB.map(r => ({ province: d.province, reseau: r.cle, couleur: r.couleur, valeur: d[r.cle] })))
        .join('rect')
        .attr('class', 'barre-gb')
        .attr('x', d => x1(d.reseau))
        .attr('width', x1.bandwidth())
        .attr('y', d => d.valeur == null ? hauteur : y(d.valeur))
        .attr('height', d => d.valeur == null ? 0 : hauteur - y(d.valeur))
        .attr('rx', 3)
        .attr('fill', d => d.couleur)
        .on('mouseenter focus', function (event, d) {
            appliquerEmphase(d.reseau);
            tooltip.html(`<strong>${d.province}</strong><br>${RESEAUX_GB.find(r => r.cle === d.reseau).label} : ${d.valeur != null ? d.valeur + ' %' : '—'}`)
                .style('opacity', 1).style('left', (event.pageX + 12) + 'px').style('top', (event.pageY - 12) + 'px');
        })
        .on('mouseleave blur', function () { appliquerEmphase(null); tooltip.style('opacity', 0); })
        .append('title')
        .text(d => `${d.province} — ${RESEAUX_GB.find(r => r.cle === d.reseau).label} : ${d.valeur != null ? d.valeur + ' %' : '—'}`);
}
