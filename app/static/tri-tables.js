/*
 * Rend triable toute table portant l'attribut data-table-triable : clic (ou Entrée/Espace au
 * clavier) sur un en-tete de colonne trie les lignes du corps de table selon cette colonne,
 * detection automatique numerique/alphabetique, bascule croissant/decroissant, aria-sort tenu
 * a jour pour les lecteurs d'ecran. Aucune dependance, vanilla JS.
 */
(function () {
    function valeurNumerique(texte) {
        // gere "67,7 %", "+12.3 pts", "1 234", etc. -- virgule francaise ou point, signe, espaces
        const nettoye = texte.replace(/\s/g, '').replace(',', '.').match(/-?\d+(\.\d+)?/);
        return nettoye ? parseFloat(nettoye[0]) : null;
    }

    function trierTable(table, indexColonne, ascendant) {
        const tbody = table.querySelector('tbody');
        const lignes = Array.from(tbody.querySelectorAll('tr'));

        function extraireValeur(ligne) {
            const cellule = ligne.children[indexColonne];
            return cellule ? cellule.textContent.trim() : '';
        }

        const toutesNumeriques = lignes.every(l => valeurNumerique(extraireValeur(l)) !== null);

        lignes.sort((a, b) => {
            const va = extraireValeur(a), vb = extraireValeur(b);
            let cmp;
            if (toutesNumeriques) {
                cmp = valeurNumerique(va) - valeurNumerique(vb);
            } else {
                cmp = va.localeCompare(vb, 'fr', { sensitivity: 'base' });
            }
            return ascendant ? cmp : -cmp;
        });

        lignes.forEach(l => tbody.appendChild(l));
    }

    document.addEventListener('DOMContentLoaded', function () {
        document.querySelectorAll('table[data-table-triable]').forEach(table => {
            const enTetes = table.querySelectorAll('thead th');
            enTetes.forEach((th, index) => {
                th.setAttribute('tabindex', '0');
                // pas de role="button" : ecraserait le role columnheader implicite du <th>, ce
                // qui invalide aria-sort (audit axe-core, regle aria-allowed-attr, 2026-08-17) et
                // supprime l'annonce d'en-tete de colonne aux lecteurs d'ecran. L'activation
                // clavier (Entree/Espace) est geree explicitement ci-dessous, sans dependre du role.
                th.setAttribute('aria-sort', 'none');
                th.style.cursor = 'pointer';
                th.title = 'Trier par ' + th.textContent.trim();

                function activer() {
                    const ascendantActuel = th.getAttribute('aria-sort') === 'ascending';
                    const nouvelOrdre = !ascendantActuel;
                    enTetes.forEach(autre => autre.setAttribute('aria-sort', 'none'));
                    th.setAttribute('aria-sort', nouvelOrdre ? 'ascending' : 'descending');
                    trierTable(table, index, nouvelOrdre);
                }

                th.addEventListener('click', activer);
                th.addEventListener('keydown', function (e) {
                    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activer(); }
                });
            });
        });
    });
})();
