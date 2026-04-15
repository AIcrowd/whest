import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '../lib/utils.js';

function ExplorerMetricCard({
  label,
  title,
  value,
  detail,
  className,
  valueClassName,
  detailClassName,
  ...props
}) {
  const heading = label ?? title;

  return (
    <Card className={className} size="sm" {...props}>
      <CardHeader className="pb-0">
        {heading ? <CardDescription className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">{heading}</CardDescription> : null}
        <CardTitle className={cn('font-mono text-3xl font-bold', valueClassName)}>
          {value}
        </CardTitle>
      </CardHeader>
      {detail ? (
        <CardContent className="pt-2">
          <div className={cn('text-sm leading-6 text-muted-foreground', detailClassName)}>{detail}</div>
        </CardContent>
      ) : null}
    </Card>
  );
}

export { ExplorerMetricCard };
export default ExplorerMetricCard;
