/*
 * Bascule manuelle clair/sombre, independante de prefers-color-scheme. Le choix est persiste en
 * localStorage et lu tres tot dans <head> (voir base.html) pour eviter un flash du mauvais theme.
 */
(function () {
    function themeEffectif() {
        var explicite = document.documentElement.getAttribute('data-theme');
        if (explicite === 'light' || explicite === 'dark') { return explicite; }
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }

    function majBouton(bouton) {
        var effectif = themeEffectif();
        bouton.setAttribute('aria-pressed', effectif === 'dark' ? 'true' : 'false');
        bouton.querySelector('.etiquette-theme').textContent = effectif === 'dark' ? 'Mode sombre' : 'Mode clair';
    }

    document.addEventListener('DOMContentLoaded', function () {
        var bouton = document.getElementById('bascule-theme');
        if (!bouton) { return; }
        majBouton(bouton);
        bouton.addEventListener('click', function () {
            var nouveau = themeEffectif() === 'dark' ? 'light' : 'dark';
            document.documentElement.setAttribute('data-theme', nouveau);
            try { localStorage.setItem('theme', nouveau); } catch (e) {}
            majBouton(bouton);
        });
    });
})();
