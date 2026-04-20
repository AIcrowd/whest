import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '../lib/utils.js';

function ExplorerMetricCard({
  label,
  title,
  value,
  detail,
  emphasis = 'default',
  className,
  valueClassName,
  detailClassName,
  ...props
}) {
  const heading = label ?? title;
  const isHero = emphasis === 'hero';

  return (
    <Card className={cn(isHero && 'rounded-lg shadow-sm', className)} size={isHero ? 'default' : 'sm'} {...props}>
      <CardHeader className={cn(isHero ? 'pb-1' : 'pb-0')}>
        {heading ? (
          <CardDescription
            className={cn(
              'text-[10px] font-semibold uppercase tracking-[0.2em]',
              isHero ? 'text-gray-600' : 'text-gray-400',
            )}
          >
            {heading}
          </CardDescription>
        ) : null}
        <CardTitle
          className={cn(
            'font-mono font-bold',
            isHero ? 'text-[2.55rem] leading-none tracking-tight sm:text-[2.9rem]' : 'text-3xl',
            valueClassName,
          )}
        >
          {value}
        </CardTitle>
      </CardHeader>
      {detail ? (
        <CardContent className={cn(isHero ? 'pt-3' : 'pt-2')}>
          <div
            className={cn(
              isHero ? 'text-[12px] font-medium leading-5 text-gray-500' : 'text-sm leading-6 text-muted-foreground',
              detailClassName,
            )}
          >
            {detail}
          </div>
        </CardContent>
      ) : null}
    </Card>
  );
}

export { ExplorerMetricCard };
export default ExplorerMetricCard;
