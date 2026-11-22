from common import parser, run
if __name__ == '__main__':
    p = parser('Wasserstein DRO sensitivity: epsilon, K=20/30, both mean modes and lookbacks')
    p.set_defaults(lookbacks=[252, 504], time_limit=30., run_name='k20_30_252_504_30s')
    run('03_dro', p.parse_args())
