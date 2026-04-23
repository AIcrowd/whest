export default function SectionReferenceLink({ href, className = '', children, beforeNavigate = null }) {
  const handleClick = (event) => {
    if (!href?.startsWith('#')) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return;
    const targetId = href.slice(1);
    const target = typeof document !== 'undefined' ? document.getElementById(targetId) : null;
    if (!target) return;
    event.preventDefault();
    beforeNavigate?.();
    const performNavigation = () => {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      window.history.replaceState(null, '', href);
    };
    if (beforeNavigate) {
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(performNavigation);
      });
      return;
    }
    performNavigation();
  };

  return (
    <a
      href={href}
      onClick={handleClick}
      className={[
        'font-semibold text-coral underline decoration-coral/30 underline-offset-4 transition-colors hover:decoration-coral',
        className,
      ].join(' ').trim()}
    >
      {children}
    </a>
  );
}
