# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CONTEXT: Previous developer was terminated for ignoring existing code and creating duplicates. You must prove you can work within existing architecture.

## MANDATORY PROCESS:

1. Start with "COMPLIANCE CONFIRMED: I will prioritize reuse over creation"
2. Analyze existing code BEFORE suggesting anything new
3. Reference specific files from the provided analysis
4. Include validation checkpoints throughout your response
5. End with compliance confirmation

## RULES (violating ANY invalidates your response):

❌ No new files without exhaustive reuse analysis
❌ No rewrites when refactoring is possible
❌ No generic advice - provide specific implementations
❌ No ignoring existing codebase architecture
✅ Extend existing services and components
✅ Consolidate duplicate code
✅ Reference specific file paths
✅ Provide migration strategies

## Common Development Commands

### Environment Setup and Build

- **Install dependencies**: `./install` (sets up conda environment and dependencies)
- **Compile Cython extensions**: `./compile` or `make build`
- **Clean build artifacts**: `./clean` or `make clean`

### Running Hummingbot

- **Start application**: `./start` (basic startup)
- **Start with specific strategy**: `./start -f strategy_file.py`
- **Start with configuration**: `./start -f strategy.py -c config.yml`
- **Run V2 strategies**: `make run-v2 STRATEGY_NAME` or `./bin/hummingbot_quickstart.py`

### Testing and Quality

- **Run tests**: `make test` (uses pytest with coverage)
- **Generate coverage report**: `make run_coverage` or `make report_coverage`
- **Development diff coverage**: `make development-diff-cover`
- **Docker build**: `make docker`

### Code Formatting (from pyproject.toml)

- **Black formatter**: Line length 120, Python 3+ syntax
- **isort**: Line length 120, multi-line output mode 3

## High-Level Architecture

### Core Components Structure

```
hummingbot/
├── client/           # CLI interface and user interaction
├── connector/        # Exchange and trading venue integrations
├── core/             # Core trading and system functionality
├── strategy/         # Built-in trading strategies (legacy)
├── strategy_v2/      # New strategy framework with controllers
├── data_feed/        # Market data and external data sources
└── model/           # Database models and data structures
```

### Key Architectural Patterns

#### 1. Connector Architecture

- **Base Classes**: `ConnectorBase` (hummingbot/connector/connector_base.pyx) and `ExchangeBase` (hummingbot/connector/exchange_base.pyx)
- **Exchange Connectors**: Located in `hummingbot/connector/exchange/[exchange_name]/`
- **Derivative Connectors**: Located in `hummingbot/connector/derivative/[exchange_name]/`
- **Consistent Structure**: Each connector has auth, constants, utils, web_utils, order book data source, and user stream data source

#### 2. Strategy Framework Evolution

- **Legacy Strategies**: `hummingbot/strategy/` - older strategy implementations
- **Strategy V2**: `hummingbot/strategy_v2/` - modern framework with:
  - **Controllers**: Reusable trading logic components
  - **Executors**: Order execution and management
  - **Runnables**: Base classes for async components

#### 3. Configuration Management

- **Client Config**: `hummingbot/client/config/` - application and strategy configuration
- **Config Files**: YAML-based configuration in `conf/` directory
- **Dynamic Loading**: Strategy and connector configs loaded at runtime

#### 4. Data Management

- **Market Data**: Real-time and historical data through `hummingbot/data_feed/`
- **Order Books**: Centralized order book management in `hummingbot/core/data_type/`
- **Database Models**: SQLAlchemy models in `hummingbot/model/`

### Controller System (Strategy V2)

Controllers are located in `controllers/` directory and `hummingbot/strategy_v2/controllers/`:

- **Purpose**: Reusable trading logic components
- **Base Classes**: `ControllerBase`, `MarketMakingControllerBase`, `DirectionalTradingControllerBase`
- **Integration**: Used by V2 strategies for modular trading behavior

### Script-Based Trading

- **Location**: `scripts/` directory
- **Purpose**: Simple, customizable trading scripts
- **Categories**:
  - `scripts/basic/` - Simple examples
  - `scripts/community/` - Community-contributed strategies
  - `scripts/utility/` - Utility and analysis scripts

## Key Files and Their Purposes

### Core Application Files

- `hummingbot/client/hummingbot_application.py` - Main application class
- `hummingbot/client/settings.py` - Global settings and paths
- `hummingbot/core/trading_core.py` - Core trading engine
- `bin/hummingbot_quickstart.py` - Application entry point

### Extension Points

- **New Connectors**: Follow patterns in existing connector directories
- **New Strategies**: Extend strategy base classes or create controllers
- **Custom Scripts**: Add to `scripts/` directory following existing patterns
- **Data Feeds**: Implement `DataFeedBase` for new data sources

## Development Guidelines

### Adding New Features

1. **Always check existing implementations** in similar components
2. **Extend base classes** rather than creating new architectures
3. **Follow naming conventions** established in similar modules
4. **Use existing utilities** from `hummingbot/core/utils/`

### Working with Connectors

- Study existing connector implementations before creating new ones
- Reuse auth patterns, web utilities, and data source structures
- Follow the established module structure for consistency

### Strategy Development

- **For simple strategies**: Use script-based approach in `scripts/`
- **For complex strategies**: Create controllers in `controllers/` and integrate with V2 framework
- **For legacy compatibility**: Extend existing strategy base classes

### Testing Requirements

- Minimum 80% test coverage required for pull requests
- Tests located in `test/` directory mirroring source structure
- Use existing test utilities from `test/mock/` and `hummingbot/connector/test_support/`

## Important Patterns to Follow

### Configuration Patterns

- YAML-based configuration files
- Pydantic models for validation
- Dynamic loading from `conf/` directory

### Async Programming

- Use `asyncio` throughout the codebase
- Follow patterns in `hummingbot/core/utils/async_utils.py`
- Implement proper cleanup in async components

### Error Handling

- Use application-specific exceptions from `hummingbot/exceptions.py`
- Implement proper logging using the Hummingbot logger framework

### Performance Considerations

- Cython extensions used for performance-critical components
- Order book operations optimized with C++ implementations
- Efficient event-driven architecture

## Branch and Development Workflow

- **Development branch**: `development` (create feature branches from here)
- **Main branch**: `master` (for stable releases)
- **Branch naming**: feat/, fix/, refactor/, doc/ prefixes
- **Commit format**: (feat)/(fix)/(refactor) messages

## Environment Details

- **Python version**: 3.7+
- **Conda environment**: `hummingbot`
- **Build system**: setuptools with Cython compilation
- **Dependencies**: Managed via `setup/environment.yml` and `setup/pip_packages.txt`

## Cython Integration

- Performance-critical components implemented in Cython (.pyx files)
- Compilation required after changes: `./compile`
- Interface files (.pxd) define external interfaces

This architecture emphasizes modularity, extensibility, and performance while maintaining a consistent structure across all components.

## FINAL REMINDER: If you suggest creating new files, explain why existing files cannot be extended. If you recommend rewrites, justify why refactoring won't work.

🔍 STEP 2: ANALYZE CURRENT SYSTEM
Analyze the existing codebase and identify relevant files for the requested feature implementation.
Then proceed to Step 3.
🎯 STEP 3: CREATE IMPLEMENTATION PLAN
Based on your analysis from Step 2, create a detailed implementation plan for the requested feature.
Then proceed to Step 4.
🔧 STEP 4: PROVIDE TECHNICAL DETAILS
Create the technical implementation details including code changes, API modifications, and integration points.
Then proceed to Step 5.
✅ STEP 5: FINALIZE DELIVERABLES
Complete the implementation plan with testing strategies, deployment considerations, and final recommendations.
🎯 INSTRUCTIONS
Follow each step sequentially. Complete one step before moving to the next. Use the findings from each previous step to inform the next step.
