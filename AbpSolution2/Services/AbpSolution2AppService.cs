using Volo.Abp.Application.Services;
using AbpSolution2.Localization;

namespace AbpSolution2.Services;

/* Inherit your application services from this class. */
public abstract class AbpSolution2AppService : ApplicationService
{
    protected AbpSolution2AppService()
    {
        LocalizationResource = typeof(AbpSolution2Resource);
    }
}