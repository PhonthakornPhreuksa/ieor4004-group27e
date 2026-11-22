from common import parser, run
if __name__ == '__main__':
    p = parser('Budgeted robust MAD: Gamma grid, K=20/30 and two mean modes')
    p.set_defaults(lookbacks=[252], time_limit=30., run_name='k20_30_252_30s')
    run('02_robust', p.parse_args())
