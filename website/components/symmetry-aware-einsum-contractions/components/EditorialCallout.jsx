const EDITORIAL_CALLOUT_SHELL_CLASS = 'rounded-2xl border border-primary/20 bg-accent/40 px-5 py-5';
const EDITORIAL_CALLOUT_KICKER_CLASS = 'font-sans text-[11px] font-semibold uppercase tracking-[0.16em] text-coral';
const EDITORIAL_CALLOUT_JUSTIFIED_STYLE = { textAlign: 'justify' };

export default function EditorialCallout({
  id,
  label,
  title,
  children,
  footer = null,
  className = '',
  labelClassName = EDITORIAL_CALLOUT_KICKER_CLASS,
  titleClassName = 'mt-1 font-heading text-base font-semibold text-foreground',
  bodyClassName = 'mt-2',
  bodyStyle = EDITORIAL_CALLOUT_JUSTIFIED_STYLE,
  footerClassName = 'mt-3 text-[13px] italic leading-6 text-stone-600',
  footerStyle = EDITORIAL_CALLOUT_JUSTIFIED_STYLE,
}) {
  return (
    <div id={id} className={[EDITORIAL_CALLOUT_SHELL_CLASS, className].join(' ').trim()}>
      {label ? (
        <div className={labelClassName}>
          {label}
        </div>
      ) : null}
      {title ? (
        <div className={titleClassName}>
          {title}
        </div>
      ) : null}
      <div className={bodyClassName} style={bodyStyle}>
        {children}
      </div>
      {footer ? (
        <div className={footerClassName} style={footerStyle}>
          {footer}
        </div>
      ) : null}
    </div>
  );
}
