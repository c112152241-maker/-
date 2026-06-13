using Volo.Abp.DependencyInjection;
using Microsoft.EntityFrameworkCore;

namespace AbpSolution2.Data;

public class AbpSolution2DbSchemaMigrator : ITransientDependency
{
    private readonly IServiceProvider _serviceProvider;

    public AbpSolution2DbSchemaMigrator(
        IServiceProvider serviceProvider)
    {
        _serviceProvider = serviceProvider;
    }

    public async Task MigrateAsync()
    {
        
        /* We intentionally resolving the AbpSolution2DbContext
         * from IServiceProvider (instead of directly injecting it)
         * to properly get the connection string of the current tenant in the
         * current scope.
         */

        await _serviceProvider
            .GetRequiredService<AbpSolution2DbContext>()
            .Database
            .MigrateAsync();

    }
}
