// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FlashLoanArbitrage
 * @notice Self-contained flash loan arbitrage contract for Aave V3
 * @dev Deploy this contract, then set FLASH_LOAN_RECEIVER_CONTRACT in ClaudeDex .env
 *
 * Deployment:
 * 1. Open https://remix.ethereum.org
 * 2. Create new file, paste this entire code
 * 3. Compile with Solidity 0.8.20
 * 4. Deploy with constructor param: 0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e (Mainnet)
 * 5. Copy deployed contract address to .env as FLASH_LOAN_RECEIVER_CONTRACT
 */

// ============ INTERFACES (Aave V3) ============

interface IPoolAddressesProvider {
    function getPool() external view returns (address);
}

interface IPool {
    function flashLoanSimple(
        address receiverAddress,
        address asset,
        uint256 amount,
        bytes calldata params,
        uint16 referralCode
    ) external;

    function FLASHLOAN_PREMIUM_TOTAL() external view returns (uint128);
}

// ============ INTERFACES (Uniswap V2 style) ============

interface IUniswapV2Router {
    function swapExactTokensForTokens(
        uint amountIn,
        uint amountOutMin,
        address[] calldata path,
        address to,
        uint deadline
    ) external returns (uint[] memory amounts);

    function getAmountsOut(uint amountIn, address[] calldata path)
        external view returns (uint[] memory amounts);
}

// ============ INTERFACES (ERC20) ============

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

// ============ MAIN CONTRACT ============

contract FlashLoanArbitrage {

    // Aave V3 Pool
    IPoolAddressesProvider public immutable ADDRESSES_PROVIDER;
    IPool public immutable POOL;

    // Owner (your wallet)
    address public owner;

    // DEX Routers (Ethereum Mainnet)
    address public constant UNISWAP_V2_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
    address public constant SUSHISWAP_ROUTER = 0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F;

    // Common tokens (Mainnet) - pass these as parameters to executeArbitrage()
    // WETH: 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2
    // USDC: 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48
    // USDT: 0xdAC17F958D2ee523a2206206994597C13D831ec7
    // DAI:  0x6B175474e89094C44Da98b954EedeACB80656c63

    // Events
    event ArbitrageExecuted(
        address indexed asset,
        uint256 amountBorrowed,
        uint256 profit,
        address buyDex,
        address sellDex
    );

    event FlashLoanReceived(
        address indexed asset,
        uint256 amount,
        uint256 premium
    );

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyPool() {
        require(msg.sender == address(POOL), "Caller must be Pool");
        _;
    }

    /**
     * @notice Constructor
     * @param _addressProvider Aave V3 PoolAddressesProvider
     *        Mainnet: 0x2f39d218133AFaB8F2B819B1066c7E434Ad94E9e
     *        Sepolia: 0x012bAC54348C0E635dCAc9D5FB99f06F24136C9A
     */
    constructor(address _addressProvider) {
        ADDRESSES_PROVIDER = IPoolAddressesProvider(_addressProvider);
        POOL = IPool(ADDRESSES_PROVIDER.getPool());
        owner = msg.sender;
    }

    /**
     * @notice Execute flash loan arbitrage
     * @param asset Token to borrow (e.g., WETH)
     * @param amount Amount to borrow in wei
     * @param buyRouter DEX with lower price
     * @param sellRouter DEX with higher price
     * @param intermediateToken Token to swap through
     */
    function executeArbitrage(
        address asset,
        uint256 amount,
        address buyRouter,
        address sellRouter,
        address intermediateToken
    ) external onlyOwner {
        require(amount > 0, "Amount must be > 0");
        require(buyRouter != sellRouter, "Routers must be different");

        // Encode params for executeOperation callback
        bytes memory params = abi.encode(
            buyRouter,
            sellRouter,
            intermediateToken
        );

        // Request flash loan - Aave will call executeOperation
        POOL.flashLoanSimple(
            address(this),  // receiverAddress (this contract)
            asset,          // asset to borrow
            amount,         // amount to borrow
            params,         // data passed to executeOperation
            0               // referralCode
        );
    }

    /**
     * @notice Aave flash loan callback - DO NOT CALL DIRECTLY
     * @dev Called by Aave Pool after sending flash loan funds
     */
    function executeOperation(
        address asset,
        uint256 amount,
        uint256 premium,
        address initiator,
        bytes calldata params
    ) external onlyPool returns (bool) {
        require(initiator == address(this), "Invalid initiator");

        emit FlashLoanReceived(asset, amount, premium);

        // Decode arbitrage parameters
        (
            address buyRouter,
            address sellRouter,
            address intermediateToken
        ) = abi.decode(params, (address, address, address));

        // ========== ARBITRAGE LOGIC ==========

        // Step 1: Swap borrowed asset -> intermediate token on cheaper DEX
        uint256 intermediateAmount = _swap(
            buyRouter,
            asset,
            intermediateToken,
            amount
        );

        // Step 2: Swap intermediate token -> asset on more expensive DEX
        uint256 finalAmount = _swap(
            sellRouter,
            intermediateToken,
            asset,
            intermediateAmount
        );

        // ========== REPAYMENT ==========

        uint256 amountOwed = amount + premium;

        // CRITICAL: Revert if not profitable (protects against sandwich attacks)
        require(finalAmount >= amountOwed, "Arbitrage not profitable after fees");

        // Approve Aave Pool to pull the owed amount
        IERC20(asset).approve(address(POOL), amountOwed);

        // Calculate and emit profit
        uint256 profit = finalAmount - amountOwed;
        emit ArbitrageExecuted(asset, amount, profit, buyRouter, sellRouter);

        return true;
    }

    /**
     * @notice Internal swap function
     */
    function _swap(
        address router,
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) internal returns (uint256 amountOut) {
        require(amountIn > 0, "Swap amount must be > 0");

        // Approve router to spend tokens
        IERC20(tokenIn).approve(router, amountIn);

        // Build swap path
        address[] memory path = new address[](2);
        path[0] = tokenIn;
        path[1] = tokenOut;

        // Execute swap
        uint[] memory amounts = IUniswapV2Router(router).swapExactTokensForTokens(
            amountIn,
            0,  // Accept any output (MEV protection via Flashbots in Python code)
            path,
            address(this),
            block.timestamp + 300  // 5 min deadline
        );

        return amounts[amounts.length - 1];
    }

    /**
     * @notice Check potential arbitrage profit (view function for simulation)
     * @return profit Expected profit in asset units (may differ from actual due to slippage)
     */
    function checkArbitrage(
        address asset,
        uint256 amount,
        address buyRouter,
        address sellRouter,
        address intermediateToken
    ) external view returns (int256 profit) {
        // Get amounts for buy leg
        address[] memory buyPath = new address[](2);
        buyPath[0] = asset;
        buyPath[1] = intermediateToken;

        uint[] memory buyAmounts;
        try IUniswapV2Router(buyRouter).getAmountsOut(amount, buyPath) returns (uint[] memory _amounts) {
            buyAmounts = _amounts;
        } catch {
            return -1; // No liquidity on buy DEX
        }

        // Get amounts for sell leg
        address[] memory sellPath = new address[](2);
        sellPath[0] = intermediateToken;
        sellPath[1] = asset;

        uint[] memory sellAmounts;
        try IUniswapV2Router(sellRouter).getAmountsOut(buyAmounts[1], sellPath) returns (uint[] memory _amounts) {
            sellAmounts = _amounts;
        } catch {
            return -2; // No liquidity on sell DEX
        }

        // Calculate profit (considering 0.05% Aave fee)
        uint256 premium = (amount * 5) / 10000;  // 0.05% fee
        uint256 amountOwed = amount + premium;

        return int256(sellAmounts[1]) - int256(amountOwed);
    }

    // ============ ADMIN FUNCTIONS ============

    /**
     * @notice Withdraw accumulated profits
     */
    function withdrawToken(address token, uint256 amount) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(balance >= amount, "Insufficient balance");
        IERC20(token).transfer(owner, amount);
    }

    /**
     * @notice Withdraw all of a token
     */
    function withdrawAllToken(address token) external onlyOwner {
        uint256 balance = IERC20(token).balanceOf(address(this));
        require(balance > 0, "No balance");
        IERC20(token).transfer(owner, balance);
    }

    /**
     * @notice Withdraw ETH using call (recommended over transfer)
     */
    function withdrawETH() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "No ETH balance");
        (bool success, ) = payable(owner).call{value: balance}("");
        require(success, "ETH transfer failed");
    }

    /**
     * @notice Transfer ownership
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }

    /**
     * @notice Get flash loan premium rate
     */
    function getFlashLoanPremium() external view returns (uint128) {
        return POOL.FLASHLOAN_PREMIUM_TOTAL();
    }

    // Allow contract to receive ETH
    receive() external payable {}
}
