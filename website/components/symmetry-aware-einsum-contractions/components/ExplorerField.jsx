import { Input } from '@/components/ui/input';
import { cn } from '../lib/utils.js';

function ExplorerField({
  label,
  hint,
  error,
  className,
  labelClassName,
  inputClassName,
  id,
  ...props
}) {
  const inputId = id ?? props.name;

  return (
    <label className={cn('grid gap-2', className)} htmlFor={inputId}>
      {label ? (
        <span className={cn('text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground', labelClassName)}>
          {label}
        </span>
      ) : null}
      <Input id={inputId} aria-invalid={error ? 'true' : undefined} className={cn('text-sm', inputClassName)} {...props} />
      {hint ? <span className="text-xs text-muted-foreground">{hint}</span> : null}
      {error ? <span className="text-xs text-destructive">{error}</span> : null}
    </label>
  );
}

export { ExplorerField };
export default ExplorerField;
